//! Fixed-density XC diagnostic; no SCF or eigenvalue computation.
use std::{env, error::Error, fs};

use muffintin::CheckpointPhysics;
use muffintin_core::{Bohr, InverseBohr};
use muffintin_coulomb::{
    CoulombRequest, EwaldScan, WeinertHartreeSpec, converged_ewald_point_kernel,
};
use muffintin_dft::{
    ElectrostaticSpec, NoncollinearXcRoute, XcFunctional, evaluate_regional_electrostatics,
    evaluate_regional_xc, xc_spec_for_density,
};
use muffintin_io::{CheckpointFile, checkpoint_file_from_toml};
use muffintin_prodbasis::TransferQ;

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("expected restart checkpoint path")?;
    let CheckpointFile::V2(checkpoint) = checkpoint_file_from_toml(&fs::read_to_string(path)?)?
    else {
        return Err("expected V2 restart checkpoint".into());
    };
    let physics = CheckpointPhysics::new(&checkpoint)?;
    let density = physics
        .restart_density()
        .ok_or("expected restart density")?;
    let nuclear_charges = checkpoint
        .geometry
        .sites
        .iter()
        .map(|site| f64::from(site.atomic_number))
        .collect::<Vec<_>>();
    let electrostatic = evaluate_regional_electrostatics(
        density.charge(),
        &ElectrostaticSpec::new(WeinertHartreeSpec::electronic(4)?, nuclear_charges.clone())?,
    )?;
    let box_size = checkpoint.geometry.lattice.vectors[0][0];
    let ewald_request = CoulombRequest::cubic(box_size, 0)?;
    let q0 = TransferQ::from_cartesian([InverseBohr(0.0); 3])?;
    let positions = checkpoint
        .geometry
        .sites
        .iter()
        .map(|site| {
            site.fractional_position
                .map(|coordinate| Bohr(coordinate * box_size))
        })
        .collect::<Vec<_>>();
    let mut ewald_nuclear_nuclear = 0.0;
    for (&left_charge, &left) in nuclear_charges.iter().zip(&positions) {
        for (&right_charge, &right) in nuclear_charges.iter().zip(&positions) {
            let kernel = converged_ewald_point_kernel(
                ewald_request.cell(),
                ewald_request.reciprocal(),
                q0,
                left,
                right,
                EwaldScan {
                    tolerance: 1.0e-6,
                    max_steps: 8,
                },
            )?;
            ewald_nuclear_nuclear += 0.5 * left_charge * right_charge * kernel.value.re;
        }
    }
    let sqrt_four_pi = (4.0 * std::f64::consts::PI).sqrt();
    let reciprocal_electron_nuclear = nuclear_charges
        .iter()
        .zip(electrostatic.raw_hartree.muffin_tins())
        .map(|(&charge, potential)| {
            -charge * potential.channel(0, 0).unwrap()[0].as_complex().re / sqrt_four_pi
        })
        .sum::<f64>();
    let nuclear_background = electrostatic.raw_nuclear.neutralizing_background_density();
    let mut candidate_background_residual = 0.0;
    for ((&charge, sphere), muffin_tin) in nuclear_charges
        .iter()
        .zip(density.geometry().spheres())
        .zip(electrostatic.weinert_density.muffin_tins())
    {
        let radius = sphere.radius.get();
        let background = |r: f64| {
            2.0 * std::f64::consts::PI / 3.0 * nuclear_background * (radius * radius - r * r)
        };
        let integrand = muffin_tin
            .channel(0, 0)
            .unwrap()
            .iter()
            .zip(muffin_tin.mesh().radii())
            .map(|(&rho, radius)| {
                sqrt_four_pi * rho.re * background(radius.get()) * radius.get().powi(2)
            })
            .collect::<Vec<_>>();
        candidate_background_residual += 0.5 * muffin_tin.mesh().integrate(&integrand)?;
        candidate_background_residual -= 0.5 * charge * background(0.0);
    }
    println!(
        "electrostatic_ha electron_hartree={:.16e} electron_nuclear_density={:.16e} electron_nuclear_nuclei={:.16e} reciprocity_residual={:.16e} nuclear_nuclear={:.16e} ewald_nuclear_nuclear={:.16e} nuclear_ewald_residual={:.16e} madelung={:.16e} coulomb={:.16e} candidate_background_residual={:.16e}",
        electrostatic.electron_hartree.get(),
        electrostatic.electron_nuclear.get(),
        reciprocal_electron_nuclear,
        electrostatic.electron_nuclear.get() - reciprocal_electron_nuclear,
        electrostatic.nuclear_nuclear.get(),
        ewald_nuclear_nuclear,
        electrostatic.nuclear_nuclear.get() - ewald_nuclear_nuclear,
        electrostatic.madelung.get(),
        electrostatic.coulomb.get(),
        candidate_background_residual,
    );
    let base = xc_spec_for_density(&density, 8, NoncollinearXcRoute::LocalSpinFrame);
    for factor in [1, 2] {
        let mut spec = base;
        spec.interstitial_divisions = base.interstitial_divisions.map(|n| n * factor);
        let xc = evaluate_regional_xc(XcFunctional::LdaPw92, &density, spec)?;
        println!(
            "divisions={:?} exc_ha={:.16e} direct_rho_vxc_ha={:.16e}",
            spec.interstitial_divisions,
            xc.exchange_correlation_energy.get(),
            xc.density_potential_integral.get(),
        );
        // This regional metric applies the interstitial mask a second time.
        // It is a representation diagnostic, NOT the band double-count term:
        // LAPW consumes the already masked potential Fourier coefficients directly.
        let projected = density
            .charge()
            .physical_inner_product(xc.potential.scalar())?;
        println!("twice_masked_rho_vxc_ha={projected:.16e}");
    }
    Ok(())
}
