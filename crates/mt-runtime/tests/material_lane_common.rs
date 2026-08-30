//! Shared Sm/Dy material harness.
//!
//! Dy and Sm lanes include this file with `#[path = "material_lane_common.rs"]`.
//! Do not copy it. SPEX HDF is frozen fields only; Checkpoint V2 comes from
//! `materialize_checkpoint_v2` plus a caller-owned signed-kappa recipe.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use muffintin::{
    CheckpointPhysics, CheckpointPhysicsError, SpinorProductInput, SpinorThcError, SpinorThcSpec,
    ThcCandidates, ThcEngine, ThcParentGrid, build_spinor_thc,
};
use muffintin_core::InverseBohr;
use muffintin_dft::{ScfConfig, ScfRelativity};
use muffintin_io::{
    CheckpointFile, CheckpointV2, SpexMaterialBasisRecipeV1, checkpoint_file_from_toml,
    materialize_checkpoint_v2, read_spex_snapshot_hdf,
};
use muffintin_prodbasis::thc::RankPolicy;

/// Provenance for one honest Checkpoint V2 consumed by a material lane.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterialProvenance {
    pub checkpoint_path: PathBuf,
    pub checkpoint_sha256: String,
    pub producer: String,
}

/// Frozen full-first-variation material fixture.
#[allow(missing_debug_implementations)]
pub struct MaterialFixture {
    pub checkpoint: CheckpointV2,
    pub config: ScfConfig,
    pub physics: CheckpointPhysics,
    pub provenance: MaterialProvenance,
}

/// Shared harness error.
#[derive(Debug)]
pub enum MaterialLaneError {
    Io(std::io::Error),
    Checkpoint(muffintin_io::IoError),
    CheckpointPhysics(CheckpointPhysicsError),
    MissingCheckpoint(PathBuf),
    NotV2,
    Relativity,
}

impl From<std::io::Error> for MaterialLaneError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<muffintin_io::IoError> for MaterialLaneError {
    fn from(error: muffintin_io::IoError) -> Self {
        Self::Checkpoint(error)
    }
}

impl From<CheckpointPhysicsError> for MaterialLaneError {
    fn from(error: CheckpointPhysicsError) -> Self {
        Self::CheckpointPhysics(error)
    }
}

/// Reject scalar Koelling–Harmon and SOC second variation.
pub fn require_spinor_first_variation(config: &ScfConfig) -> Result<(), MaterialLaneError> {
    match config.relativity {
        ScfRelativity::SpinorFirstVariation => Ok(()),
        ScfRelativity::Scalar | ScfRelativity::SocSecondVariation { .. } => {
            Err(MaterialLaneError::Relativity)
        }
    }
}

/// Load a Checkpoint V2 and bind a spinor-first-variation config.
///
/// `checkpoint_sha256` is caller-recorded; this helper does not hash the file.
pub fn load_spinor_checkpoint_v2(
    path: &Path,
    config: ScfConfig,
    provenance: MaterialProvenance,
) -> Result<MaterialFixture, MaterialLaneError> {
    require_spinor_first_variation(&config)?;
    if !path.is_file() {
        return Err(MaterialLaneError::MissingCheckpoint(path.to_path_buf()));
    }
    let text = std::fs::read_to_string(path)?;
    let CheckpointFile::V2(checkpoint) = checkpoint_file_from_toml(&text)? else {
        return Err(MaterialLaneError::NotV2);
    };
    let physics = CheckpointPhysics::new(&checkpoint)?;
    Ok(MaterialFixture {
        checkpoint,
        config,
        physics,
        provenance,
    })
}

/// Load SPEX frozen fields, apply a caller-owned signed-kappa recipe, then bind.
pub fn load_spex_material(
    spex_path: &Path,
    recipe: &SpexMaterialBasisRecipeV1,
    config: ScfConfig,
    provenance: MaterialProvenance,
) -> Result<MaterialFixture, MaterialLaneError> {
    require_spinor_first_variation(&config)?;
    if !spex_path.is_file() {
        return Err(MaterialLaneError::MissingCheckpoint(
            spex_path.to_path_buf(),
        ));
    }
    let fields = read_spex_snapshot_hdf(spex_path)?;
    let materialized = materialize_checkpoint_v2(&fields, recipe)?;
    let physics =
        CheckpointPhysics::new_spex_material(&materialized.checkpoint, recipe, &config.basis)?;
    Ok(MaterialFixture {
        checkpoint: materialized.checkpoint,
        config,
        physics,
        provenance,
    })
}

/// Complete k-mesh q slice in production index order.
///
/// `q_index i` uses `k_fractional[i]` so `require_spinor_q_slice` holds.
/// Finite-q wrap lives on [`muffintin::SpinorKMinusQ`], not `TransferQ::umklapp`.
pub fn ordered_q_slice(
    fixture: &MaterialFixture,
) -> Result<Vec<SpinorProductInput>, CheckpointPhysicsError> {
    let seed = fixture
        .physics
        .spinor_product_input(&fixture.config, [0.0; 3])?;
    let k_fractional = seed.orbitals.k_fractional.clone();
    k_fractional
        .into_iter()
        .map(|q_frac| {
            fixture
                .physics
                .spinor_product_input(&fixture.config, q_frac)
        })
        .collect()
}

/// QRCP and pivoted Cholesky on the same parent grid, candidates, and rank.
///
/// Compare residual / action / quadratic forms. Do not require pivot or
/// zeta elementwise equality.
pub fn compare_qrcp_cholesky(
    inputs: &[SpinorProductInput],
    grid: &ThcParentGrid,
    rank: RankPolicy,
    candidates: ThcCandidates,
) -> Result<(muffintin::SpinorThcResult, muffintin::SpinorThcResult), SpinorThcError> {
    let qrcp = build_spinor_thc(
        inputs,
        grid,
        &SpinorThcSpec {
            rank,
            candidates: candidates.clone(),
            engine: ThcEngine::FullColumnPivotedQr,
        },
    )?;
    let cholesky = build_spinor_thc(
        inputs,
        grid,
        &SpinorThcSpec {
            rank,
            candidates,
            engine: ThcEngine::FullPivotedCholesky,
        },
    )?;
    Ok((qrcp, cholesky))
}

#[cfg(test)]
mod tests {
    use muffintin_core::Hartree;
    use muffintin_dft::{
        NoncollinearXcRoute, ScfBasis, ScfConvergence, ScfExchangeCorrelation, ScfKMesh, ScfMixing,
        ScfOccupations, XcFunctional,
    };

    use super::*;

    fn dummy_config(relativity: ScfRelativity) -> ScfConfig {
        ScfConfig {
            electron_count: 1.0,
            k_mesh: ScfKMesh {
                divisions: [1, 1, 1],
                shift: [0.0; 3],
            },
            basis: ScfBasis {
                plane_wave_cutoff: InverseBohr(1.0),
                l_max: 1,
                channels: Vec::new(),
                resolved_channels: Vec::new(),
            },
            occupations: ScfOccupations::FermiDirac {
                temperature: Hartree(0.02),
            },
            exchange_correlation: ScfExchangeCorrelation {
                functional: XcFunctional::LdaPw92,
                noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
            },
            mixing: ScfMixing::Linear { alpha: 1.0 },
            relativity,
            convergence: ScfConvergence {
                energy_tolerance: Hartree(1.0),
                density_tolerance: 1.0,
                max_iterations: 1,
            },
            core_sites: Vec::new(),
        }
    }

    #[test]
    fn load_rejects_missing_checkpoint() {
        let path = Path::new("/no/such/sm-fcc-checkpoint.toml");
        let provenance = MaterialProvenance {
            checkpoint_path: path.to_path_buf(),
            checkpoint_sha256: String::new(),
            producer: "absent".to_owned(),
        };
        match load_spinor_checkpoint_v2(
            path,
            dummy_config(ScfRelativity::SpinorFirstVariation),
            provenance,
        ) {
            Err(MaterialLaneError::MissingCheckpoint(missing)) => assert_eq!(missing, path),
            Err(_) => panic!("expected missing checkpoint"),
            Ok(_) => panic!("missing checkpoint path must not load"),
        }
    }

    #[test]
    fn load_rejects_scalar_relativity_before_io() {
        let path = Path::new("/no/such/sm-fcc-checkpoint.toml");
        let provenance = MaterialProvenance {
            checkpoint_path: path.to_path_buf(),
            checkpoint_sha256: String::new(),
            producer: "absent".to_owned(),
        };
        match load_spinor_checkpoint_v2(path, dummy_config(ScfRelativity::Scalar), provenance) {
            Err(MaterialLaneError::Relativity) => {}
            Err(_) => panic!("expected relativity reject"),
            Ok(_) => panic!("scalar relativity must not load"),
        }
    }
}
