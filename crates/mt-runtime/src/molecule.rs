//! High-level periodic molecule input lowered to the neutral atomic start.

use std::collections::BTreeMap;

use muffintin_core::{AngularGrid, Bohr, ExponentialMesh, InverseBohr, SPEX_SPEED_OF_LIGHT};
use muffintin_dft::{FreeAtomScfSpec, ScfExchangeCorrelation, XcFunctional};
use muffintin_io::{
    AngularBasis, CheckpointMeta, EnergyUnit, ExponentialMeshSpec, GeometryV2, LatticeV1,
    LinearizationV1, PotentialConventionV1, PotentialRadialQuantityV1, RadialBasisSpinV2,
    RadialEquationTag, SiteRadialBasisV2, SiteV2, SphericalChannelConvention,
};
use thiserror::Error;

use crate::checkpoint_physics::{
    AtomicStartError, AtomicStartRequest, RegionalFieldLayout, Structure, materialize_atomic_start,
};
use crate::input::{
    ExchangeCorrelation, Input, MoleculeAtom, MoleculeCellShape, MoleculeInput, Relativity, Task,
};

/// Failure lowering a validated molecule source into an atomic-start checkpoint.
#[derive(Debug, Error)]
pub enum MoleculeStartError {
    #[error("molecule source coordinates overflowed the generated cell")]
    CellOverflow,
    #[error("molecule vacuum must be greater than the muffin-tin radius")]
    VacuumTooSmall,
    #[error("molecule radial mesh increment is not finite and positive")]
    InvalidRadialMesh,
    #[error("molecule free-atom mesh does not reach the muffin-tin radius")]
    FreeAtomMeshTooShort,
    #[error("molecule source did not contain a dft-scf task")]
    MissingScfTask,
    #[error(transparent)]
    Mesh(#[from] muffintin_core::MeshError),
    #[error(transparent)]
    Grid(#[from] muffintin_core::GridError),
    #[error(transparent)]
    FieldLayout(#[from] crate::RegionalFieldLayoutError),
    #[error(transparent)]
    Structure(#[from] crate::CheckpointPhysicsError),
    #[error(transparent)]
    AtomicStart(#[from] AtomicStartError),
}

/// Lower a molecule source to the same V2 restart consumed by the checkpoint path.
pub(crate) fn materialize_molecule_input(
    input: &Input,
    molecule: &MoleculeInput,
) -> Result<muffintin_io::CheckpointV2, MoleculeStartError> {
    let Task::DftScf { xc, relativity, .. } = input
        .workflow
        .tasks
        .iter()
        .filter_map(|task_id| input.task.get(task_id))
        .find(|task| matches!(task, Task::DftScf { .. }))
        .ok_or(MoleculeStartError::MissingScfTask)?
    else {
        unreachable!("the molecule source validator requires one dft-scf task")
    };

    let speed_of_light = input.speed_of_light.unwrap_or(SPEX_SPEED_OF_LIGHT);
    let geometry = build_geometry(molecule, *relativity)?;
    let structure = Structure::new(geometry)?;
    let field_layout = RegionalFieldLayout::from_g_cutoff(
        &structure,
        InverseBohr(molecule.field_g_cutoff),
        molecule.field_l_max,
    )?;
    let free_atom_mesh = ExponentialMesh::new(
        Bohr(molecule.free_atom.first),
        molecule.free_atom.log_increment,
        molecule.free_atom.points,
    )?;
    if free_atom_mesh.last().get() < molecule.muffin_tin_radius {
        return Err(MoleculeStartError::FreeAtomMeshTooShort);
    }
    let start = materialize_atomic_start(AtomicStartRequest {
        meta: molecule_meta(molecule, speed_of_light),
        structure,
        field_layout,
        exchange_correlation: map_exchange_correlation(*xc),
        free_atom_scf: FreeAtomScfSpec {
            mesh: free_atom_mesh,
            speed_of_light,
            mixing: molecule.free_atom.mixing,
            potential_tolerance: molecule.free_atom.potential_tolerance,
            tail_tolerance: molecule.free_atom.tail_tolerance,
            max_iterations: molecule.free_atom.max_iterations,
        },
        angular_grid: AngularGrid::fibonacci(molecule.angular_points)?,
    })?;
    Ok(start.checkpoint)
}

fn build_geometry(
    molecule: &MoleculeInput,
    relativity: Relativity,
) -> Result<GeometryV2, MoleculeStartError> {
    let (minimum, maximum) = bounding_box(&molecule.atoms);
    let span = std::array::from_fn(|axis| maximum[axis] - minimum[axis]);
    if span.iter().any(|value: &f64| !value.is_finite()) {
        return Err(MoleculeStartError::CellOverflow);
    }
    if molecule.vacuum <= molecule.muffin_tin_radius {
        return Err(MoleculeStartError::VacuumTooSmall);
    }
    let (lengths, offset) = match molecule.cell_shape {
        MoleculeCellShape::Orthorhombic => (
            span.map(|value| value + 2.0 * molecule.vacuum),
            minimum.map(|value| -value + molecule.vacuum),
        ),
        MoleculeCellShape::Cubic => {
            let side = span.into_iter().fold(0.0_f64, f64::max) + 2.0 * molecule.vacuum;
            (
                [side; 3],
                std::array::from_fn(|axis| -minimum[axis] + 0.5 * (side - span[axis])),
            )
        }
    };
    if lengths
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || offset.iter().any(|value| !value.is_finite())
    {
        return Err(MoleculeStartError::CellOverflow);
    }
    let fractional = molecule
        .atoms
        .iter()
        .map(|atom| {
            std::array::from_fn(|axis| (atom.position[axis] + offset[axis]) / lengths[axis])
        })
        .collect::<Vec<_>>();
    if fractional
        .iter()
        .flatten()
        .any(|value| !value.is_finite() || *value < 0.0 || *value >= 1.0)
    {
        return Err(MoleculeStartError::CellOverflow);
    }
    let radial_log_increment = (molecule.muffin_tin_radius / molecule.radial_first).ln()
        / (molecule.radial_points - 1) as f64;
    if !radial_log_increment.is_finite() || radial_log_increment <= 0.0 {
        return Err(MoleculeStartError::InvalidRadialMesh);
    }
    let radial_equation = match relativity {
        Relativity::SpinorFirstVariation {} => RadialEquationTag::FullyRelativisticDirac,
        Relativity::Scalar {} | Relativity::SocSecondVariation { .. } => {
            RadialEquationTag::ScalarKoellingHarmon
        }
    };
    let sites = molecule
        .atoms
        .iter()
        .zip(fractional)
        .map(|(atom, fractional_position)| SiteV2 {
            id: atom.id.clone(),
            atomic_number: atom.atomic_number,
            fractional_position,
            muffin_tin_radius_unit: molecule.length_unit,
            muffin_tin_radius: molecule.muffin_tin_radius,
        })
        .collect::<Vec<_>>();
    let spins = if matches!(relativity, Relativity::SpinorFirstVariation {}) {
        vec![RadialBasisSpinV2::Up, RadialBasisSpinV2::Down]
    } else {
        vec![RadialBasisSpinV2::Scalar]
    };
    let radial_basis = molecule
        .atoms
        .iter()
        .flat_map(|atom| spins.iter().copied().map(move |spin| (atom, spin)))
        .map(|(atom, spin)| SiteRadialBasisV2 {
            site_id: atom.id.clone(),
            spin,
            mesh: ExponentialMeshSpec {
                radius_unit: molecule.length_unit,
                first: molecule.radial_first,
                log_increment: radial_log_increment,
                point_count: molecule.radial_points,
                last: molecule.muffin_tin_radius,
                consistency_tolerance: 1.0e-12,
            },
            radial_equation,
            linearization: LinearizationV1 {
                energy_unit: EnergyUnit::Hartree,
                linearization_energies: Vec::new(),
                local_orbital_energies: Vec::new(),
            },
        })
        .collect();
    Ok(GeometryV2 {
        lattice: LatticeV1 {
            unit: molecule.length_unit,
            vectors: [
                [lengths[0], 0.0, 0.0],
                [0.0, lengths[1], 0.0],
                [0.0, 0.0, lengths[2]],
            ],
        },
        sites,
        radial_basis,
    })
}

fn bounding_box(atoms: &[MoleculeAtom]) -> ([f64; 3], [f64; 3]) {
    let first = atoms[0].position;
    atoms
        .iter()
        .skip(1)
        .fold((first, first), |(minimum, maximum), atom| {
            (
                std::array::from_fn(|axis| minimum[axis].min(atom.position[axis])),
                std::array::from_fn(|axis| maximum[axis].max(atom.position[axis])),
            )
        })
}

fn molecule_meta(molecule: &MoleculeInput, speed_of_light: f64) -> CheckpointMeta {
    let mut annotations = BTreeMap::from([
        ("input.source".to_owned(), "molecule".to_owned()),
        ("molecule.boundary".to_owned(), "periodic".to_owned()),
        (
            "molecule.cell_shape".to_owned(),
            match molecule.cell_shape {
                MoleculeCellShape::Orthorhombic => "orthorhombic",
                MoleculeCellShape::Cubic => "cubic",
            }
            .to_owned(),
        ),
        ("molecule.length_unit".to_owned(), "bohr".to_owned()),
        (
            "molecule.vacuum_bohr".to_owned(),
            molecule.vacuum.to_string(),
        ),
    ]);
    annotations.insert(
        "molecule.atomic_start_electron_count".to_owned(),
        molecule.neutral_electron_count().to_string(),
    );
    CheckpointMeta {
        title: "periodic molecule-in-box atomic start".to_owned(),
        producer: "libmuffintin-runtime molecule input".to_owned(),
        producer_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
        speed_of_light,
        energy_zero: "periodic molecule-in-box electrostatic reference".to_owned(),
        potential_convention: PotentialConventionV1 {
            angular_basis: AngularBasis::ComplexCondonShortley,
            radial_quantity: PotentialRadialQuantityV1::Potential,
            spherical_channel: SphericalChannelConvention::PhysicalValue,
        },
        annotations,
    }
}

fn map_exchange_correlation(xc: ExchangeCorrelation) -> ScfExchangeCorrelation {
    match xc {
        ExchangeCorrelation::LdaPw92 {
            noncollinear_route,
            interstitial_grid,
        } => ScfExchangeCorrelation {
            functional: XcFunctional::LdaPw92,
            noncollinear_route: map_noncollinear_xc_route(noncollinear_route),
            interstitial_grid,
        },
        ExchangeCorrelation::Pbe {
            noncollinear_route,
            interstitial_grid,
        } => ScfExchangeCorrelation {
            functional: XcFunctional::Pbe,
            noncollinear_route: map_noncollinear_xc_route(noncollinear_route),
            interstitial_grid,
        },
    }
}

const fn map_noncollinear_xc_route(
    route: crate::NoncollinearXcRoute,
) -> muffintin_dft::NoncollinearXcRoute {
    match route {
        crate::NoncollinearXcRoute::LocalSpinFrame => {
            muffintin_dft::NoncollinearXcRoute::LocalSpinFrame
        }
        crate::NoncollinearXcRoute::MagnetizationField => {
            muffintin_dft::NoncollinearXcRoute::MagnetizationField
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MoleculeBoundary, MoleculeFreeAtom};

    fn source() -> MoleculeInput {
        MoleculeInput {
            boundary: MoleculeBoundary::Periodic,
            cell_shape: MoleculeCellShape::Cubic,
            length_unit: muffintin_io::LengthUnit::Bohr,
            vacuum: 4.3,
            muffin_tin_radius: 0.65,
            radial_first: 1.0e-6,
            radial_points: 401,
            field_g_cutoff: 12.0,
            field_l_max: 8,
            angular_points: 302,
            free_atom: MoleculeFreeAtom {
                first: 1.0e-8,
                log_increment: 0.01,
                points: 2_143,
                mixing: 0.3,
                potential_tolerance: 2.0e-5,
                tail_tolerance: 1.0e-7,
                max_iterations: 120,
            },
            atoms: vec![
                MoleculeAtom {
                    id: "H-1".to_owned(),
                    atomic_number: 1,
                    position: [-0.7, 0.0, 0.0],
                },
                MoleculeAtom {
                    id: "H-2".to_owned(),
                    atomic_number: 1,
                    position: [0.7, 0.0, 0.0],
                },
            ],
        }
    }

    #[test]
    fn cubic_cell_centers_cartesian_molecule_with_requested_vacuum() {
        let geometry = build_geometry(&source(), Relativity::Scalar {}).unwrap();
        assert_eq!(
            geometry.lattice.vectors,
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        );
        assert!((geometry.sites[0].fractional_position[0] - 0.43).abs() < 1.0e-14);
        assert!((geometry.sites[1].fractional_position[0] - 0.57).abs() < 1.0e-14);
        for site in &geometry.sites {
            assert_eq!(site.fractional_position[1], 0.5);
            assert_eq!(site.fractional_position[2], 0.5);
        }
    }

    #[test]
    fn spinor_molecule_geometry_has_the_exact_up_down_radial_pair() {
        let geometry = build_geometry(&source(), Relativity::SpinorFirstVariation {}).unwrap();
        assert_eq!(geometry.radial_basis.len(), 4);
        assert_eq!(
            geometry
                .radial_basis
                .iter()
                .map(|basis| (basis.site_id.as_str(), basis.spin))
                .collect::<Vec<_>>(),
            vec![
                ("H-1", RadialBasisSpinV2::Up),
                ("H-1", RadialBasisSpinV2::Down),
                ("H-2", RadialBasisSpinV2::Up),
                ("H-2", RadialBasisSpinV2::Down),
            ]
        );
    }
}
