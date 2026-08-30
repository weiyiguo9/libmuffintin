//! Public spinor AllQL2 THC tests on frozen spinor product input.

use muffintin::{
    RankPolicy, SPINOR_RADIAL_LO0, SPINOR_RADIAL_P, SPINOR_RADIAL_PDOT, CheckpointPhysics,
    SpinorProductInput, SpinorThcError, SpinorThcSpec, ThcCandidates, ThcEngine, ThcParentGrid,
    ThcRegion, build_spinor_thc,
};
use muffintin_core::{
    Bohr, InverseBohr, RelativisticChannel, SpinProjection, complex_spherical_harmonics, lm_index,
};
use muffintin_operators::CompiledSiteProjection;
use muffintin_prodbasis::thc::{GridPath, L2Engine, SelectorStrategy};
use num_complex::Complex64;

#[path = "spinor_hydrogen.rs"]
mod spinor_hydrogen;

use spinor_hydrogen::{hydrogen_spinor_checkpoint, parent_grid, spinor_config};

fn spec_all(n_mu: usize) -> SpinorThcSpec {
    spec_all_engine(n_mu, ThcEngine::FullColumnPivotedQr)
}

fn spec_all_engine(n_mu: usize, engine: ThcEngine) -> SpinorThcSpec {
    SpinorThcSpec {
        rank: RankPolicy::Exact { n_mu },
        candidates: ThcCandidates::All,
        engine,
    }
}

#[derive(Clone, Copy)]
enum MtKPhase {
    CellPeriodic,
    Missing,
    Opposite,
}

#[derive(Clone, Copy)]
enum SmallAngular {
    OppositeKappa,
    SameKappa,
}

#[derive(Clone, Copy)]
enum SmallRadial {
    PhysicalQ,
    CQ,
    Skip,
}

#[derive(Clone, Copy, Default)]
struct PauliSample {
    large: [Complex64; 2],
    small: [Complex64; 2],
}

fn pauli_omega(channel: RelativisticChannel, harmonics: &[Complex64]) -> [Complex64; 2] {
    let mut pauli = [Complex64::default(); 2];
    for term in channel.spinor_harmonic_terms().into_iter().flatten() {
        let y = harmonics[lm_index(term.orbital.l, term.orbital.m).unwrap()];
        let spin = match term.spin {
            SpinProjection::Up => 0,
            SpinProjection::Down => 1,
        };
        pauli[spin] += Complex64::from(term.coefficient) * y;
    }
    pauli
}

fn site_channels(
    compiled: &muffintin_operators::lapw::SpinorCompiledBasis,
    site: usize,
) -> &[RelativisticChannel] {
    compiled.site_augmentations[site][0].channels.as_slice()
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn independent_sample(
    input: &SpinorProductInput,
    grid: &ThcParentGrid,
    point: usize,
    k: usize,
    band: usize,
    skip_pdot: bool,
    skip_lo: bool,
    small_angular: SmallAngular,
    small_radial: SmallRadial,
    mt_k_phase: MtKPhase,
    omit_spin: Option<usize>,
    conjugate_rows: bool,
    reverse_g: bool,
) -> PauliSample {
    let compiled = &input.orbitals.bases[k];
    let evecs = &input.orbitals.eigenvectors[k];
    match grid.points()[point].region {
        ThcRegion::Interstitial => {
            let volume = input
                .source
                .partition
                .interstitial()
                .cell_volume()
                .get()
                .sqrt();
            let r = grid.points()[point].coordinate;
            let mut large = [Complex64::default(); 2];
            for (g, wave) in compiled.plane_waves.iter().enumerate() {
                let argument: f64 = wave
                    .g
                    .cartesian
                    .iter()
                    .zip(r)
                    .map(|(g, x)| g.get() * x.get())
                    .sum();
                let phase =
                    Complex64::from_polar(1.0, if reverse_g { -argument } else { argument });
                for spin in 0..2 {
                    if omit_spin == Some(spin) {
                        continue;
                    }
                    let row = compiled.layout.plane_wave_index(spin, g).unwrap();
                    let mut coeff = evecs.at(row, band);
                    if conjugate_rows {
                        coeff = coeff.conj();
                    }
                    large[spin] += coeff * phase / volume;
                }
            }
            PauliSample {
                large,
                small: [Complex64::default(); 2],
            }
        }
        ThcRegion::MuffinTin { site, radial_index } => {
            let projected =
                CompiledSiteProjection::spinor(compiled, site, site_channels(compiled, site))
                    .unwrap()
                    .project_eigenvectors(evecs)
                    .unwrap();
            let radials = &input.source.radials[site];
            let origin = input.source.partition.sites()[site].position;
            let r = grid.points()[point].coordinate;
            let direction = [
                r[0].get() - origin[0].get(),
                r[1].get() - origin[1].get(),
                r[2].get() - origin[2].get(),
            ];
            let radius = radials.mesh.radii()[radial_index].get();
            let inv_r = 1.0 / radius;
            let mut l_max = 0;
            for coord in 0..projected.coordinate_count() {
                let (id, _) = input.site_projection_identity(site, coord).unwrap();
                l_max = l_max.max(id.kappa.large_l()).max(id.kappa.small_l());
            }
            let harmonics = complex_spherical_harmonics(l_max, direction);
            let mut sample = PauliSample::default();
            for coord in 0..projected.coordinate_count() {
                let (id, twice_mu) = input.site_projection_identity(site, coord).unwrap();
                if skip_pdot && id.n == SPINOR_RADIAL_PDOT {
                    continue;
                }
                if skip_lo && id.n >= SPINOR_RADIAL_LO0 {
                    continue;
                }
                let radial = radials
                    .valence
                    .iter()
                    .find(|radial| radial.kappa == id.kappa && radial.n == id.n)
                    .unwrap();
                let channel = RelativisticChannel::new(id.kappa, twice_mu).unwrap();
                let large_omega = pauli_omega(channel, &harmonics);
                let small_channel = match small_angular {
                    SmallAngular::OppositeKappa => channel.opposite_kappa(),
                    SmallAngular::SameKappa => channel,
                };
                let small_omega = pauli_omega(small_channel, &harmonics);
                let amplitude = projected.at(coord, band) * inv_r;
                let p = amplitude * radial.samples.large[radial_index];
                let mut q = amplitude * radial.samples.small[radial_index];
                if matches!(small_radial, SmallRadial::Skip) {
                    q = Complex64::default();
                }
                for spin in 0..2 {
                    sample.large[spin] += p * large_omega[spin];
                    sample.small[spin] += q * small_omega[spin];
                }
            }
            let argument: f64 = compiled.plane_waves[0]
                .k
                .iter()
                .zip(r)
                .map(|(component, point)| component.get() * point.get())
                .sum();
            let phase = match mt_k_phase {
                MtKPhase::CellPeriodic => Complex64::from_polar(1.0, -argument),
                MtKPhase::Missing => Complex64::new(1.0, 0.0),
                MtKPhase::Opposite => Complex64::from_polar(1.0, argument),
            };
            for spin in 0..2 {
                sample.large[spin] *= phase;
                sample.small[spin] *= phase;
            }
            sample
        }
    }
}

fn wrap_phase(input: &SpinorProductInput, k: usize, r: [Bohr; 3], sign: f64) -> Complex64 {
    let mapped = input
        .k_minus_q
        .iter()
        .find(|mapped| mapped.k_index == k)
        .unwrap();
    let argument = mapped
        .umklapp
        .cartesian
        .iter()
        .zip(r)
        .map(|(g, x)| g.get() * x.get())
        .sum::<f64>()
        * sign;
    Complex64::from_polar(1.0, argument)
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn independent_pair(
    input: &SpinorProductInput,
    grid: &ThcParentGrid,
    point: usize,
    k: usize,
    left: usize,
    right: usize,
    skip_pdot: bool,
    skip_lo: bool,
    small_angular: SmallAngular,
    small_radial: SmallRadial,
    phase_sign: f64,
    extra_q: bool,
    mt_k_phase: MtKPhase,
    omit_spin: Option<usize>,
    conjugate_rows: bool,
    reverse_g: bool,
) -> Complex64 {
    let mapped = input
        .k_minus_q
        .iter()
        .find(|mapped| mapped.k_index == k)
        .unwrap();
    let mut left_s = independent_sample(
        input,
        grid,
        point,
        mapped.kq_index,
        left,
        skip_pdot,
        skip_lo,
        small_angular,
        small_radial,
        mt_k_phase,
        omit_spin,
        conjugate_rows,
        reverse_g,
    );
    let right_s = independent_sample(
        input,
        grid,
        point,
        mapped.k_index,
        right,
        skip_pdot,
        skip_lo,
        small_angular,
        SmallRadial::PhysicalQ,
        mt_k_phase,
        omit_spin,
        conjugate_rows,
        reverse_g,
    );
    if matches!(small_radial, SmallRadial::CQ) {
        for spin in 0..2 {
            left_s.small[spin] *= Complex64::new(0.0, 1.0);
        }
    }
    let mut density = Complex64::default();
    for spin in 0..2 {
        density += left_s.large[spin].conj() * right_s.large[spin]
            + left_s.small[spin].conj() * right_s.small[spin];
    }
    let mut phase = wrap_phase(input, k, grid.points()[point].coordinate, phase_sign);
    if extra_q {
        let argument = input
            .source
            .q
            .umklapp
            .cartesian
            .iter()
            .zip(grid.points()[point].coordinate)
            .map(|(g, x)| g.get() * x.get())
            .sum();
        phase *= Complex64::from_polar(1.0, argument);
    }
    phase * density
}

fn selected_pair(
    result: &muffintin::SpinorThcResult,
    q: usize,
    parent: usize,
    column: usize,
) -> Complex64 {
    let mu = result
        .selection
        .points
        .iter()
        .position(|point| point.id == parent)
        .expect("parent point was selected");
    result.records[q].vertices[column].coefficients()[mu]
}

fn default_pair(
    input: &SpinorProductInput,
    grid: &ThcParentGrid,
    point: usize,
    k: usize,
) -> Complex64 {
    independent_pair(
        input,
        grid,
        point,
        k,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    )
}

#[test]
fn q0_mt_pp_qq_oracle_and_interstitial_two_pauli() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let input = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let n_parent = grid.points().len();
    let n_positive = grid
        .points()
        .iter()
        .filter(|point| point.weight > 0.0)
        .count();
    assert_eq!(n_parent, 6);
    assert_eq!(n_positive, 5);
    assert!(
        input.source.radials[0]
            .valence
            .iter()
            .any(|radial| radial.n == SPINOR_RADIAL_P
                && radial.samples.small.iter().any(|q| q.abs() > 0.0))
    );
    let result =
        build_spinor_thc(std::slice::from_ref(&input), &grid, &spec_all(n_positive)).unwrap();
    assert_eq!(result.records.len(), 1);
    assert_eq!(result.effective_rank, n_positive);
    assert_eq!(result.records[0].fit.n_points, n_parent);
    assert!(!result.selection.points.iter().any(|point| point.id == 1));
    assert_eq!(
        result.selection.provenance.grid_path,
        GridPath::External {
            n_points: n_parent,
            n_candidates: n_positive,
        }
    );
    assert_eq!(
        result.selection.provenance.strategy,
        SelectorStrategy::AllQL2
    );
    let mt = 0;
    let mt_mid = 2;
    let interstitial = 3;
    let column = input.pair_columns.encode(0, 0, 0);
    let exact_mt = default_pair(&input, &grid, mt, 0);
    let got_mt = selected_pair(&result, 0, mt, column);
    assert!(
        (got_mt - exact_mt).norm() < 1.0e-8,
        "{got_mt} vs {exact_mt}"
    );
    let exact_mid = default_pair(&input, &grid, mt_mid, 0);
    let got_mid = selected_pair(&result, 0, mt_mid, column);
    assert!(
        (got_mid - exact_mid).norm() < 1.0e-8,
        "{got_mid} vs {exact_mid}"
    );
    let without_pdot = independent_pair(
        &input,
        &grid,
        mt,
        0,
        0,
        0,
        true,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    let without_lo = independent_pair(
        &input,
        &grid,
        mt,
        0,
        0,
        0,
        false,
        true,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    assert!(
        (without_pdot - exact_mt).norm() > 1.0e-8 || (without_lo - exact_mt).norm() > 1.0e-8,
        "Pdot or signed-kappa LO/RLO must contribute to the MT pair density"
    );
    let without_qq = independent_pair(
        &input,
        &grid,
        mt_mid,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::Skip,
        1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    assert!(
        (without_qq - exact_mid).norm() > 1.0e-10,
        "physical QQ must contribute: {without_qq} vs {exact_mid}"
    );
    let exact_i = default_pair(&input, &grid, interstitial, 0);
    let got_i = selected_pair(&result, 0, interstitial, column);
    assert!((got_i - exact_i).norm() < 1.0e-8, "{got_i} vs {exact_i}");
    let conjugated = independent_pair(
        &input,
        &grid,
        interstitial,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        true,
        false,
    );
    let reversed = independent_pair(
        &input,
        &grid,
        interstitial,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        false,
        true,
    );
    let omit_down = independent_pair(
        &input,
        &grid,
        interstitial,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        false,
        MtKPhase::CellPeriodic,
        Some(1),
        false,
        false,
    );
    assert!((got_i - conjugated).norm() > 1.0e-8);
    assert!((got_i - reversed).norm() > 1.0e-8);
    assert!(
        (got_i - omit_down).norm() > 1.0e-10,
        "both Pauli PW blocks must contribute"
    );
}

#[test]
fn finite_q_uses_cell_periodic_k_phase_and_stored_wrap() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let q0 = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let q15 = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [1.5, 0.0, 0.0])
        .unwrap();
    assert_eq!(q15.source.q.umklapp.index, [1, 0, 0]);
    assert_eq!(q15.k_minus_q[0].umklapp.index, [-1, 0, 0]);
    let grid = parent_grid(&q15);
    let spec = SpinorThcSpec {
        rank: RankPolicy::Exact { n_mu: 1 },
        candidates: ThcCandidates::Indices(vec![0]),
        engine: ThcEngine::FullColumnPivotedQr,
    };
    let result = build_spinor_thc(&[q0.clone(), q15.clone()], &grid, &spec).unwrap();
    assert_eq!(result.records[1].q_index, 1);
    assert_eq!(result.records[1].q, q15.source.q);
    let parent = 0;
    let column = q15.pair_columns.encode(0, 0, 0);
    let exact = default_pair(&q15, &grid, parent, 0);
    let got = selected_pair(&result, 1, parent, column);
    assert!((got - exact).norm() < 1.0e-8, "{got} vs {exact}");
    let flipped = independent_pair(
        &q15,
        &grid,
        parent,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        -1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    let doubled = independent_pair(
        &q15,
        &grid,
        parent,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        true,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    let missing_k = independent_pair(
        &q15,
        &grid,
        parent,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        false,
        MtKPhase::Missing,
        None,
        false,
        false,
    );
    let opposite_k = independent_pair(
        &q15,
        &grid,
        parent,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        false,
        MtKPhase::Opposite,
        None,
        false,
        false,
    );
    assert!((got - flipped).norm() > 1.0e-8);
    assert!((got - doubled).norm() > 1.0e-8);
    assert!((got - missing_k).norm() > 1.0e-8);
    assert!((got - opposite_k).norm() > 1.0e-8);

    let spec_mid = SpinorThcSpec {
        rank: RankPolicy::Exact { n_mu: 1 },
        candidates: ThcCandidates::Indices(vec![2]),
        engine: ThcEngine::FullColumnPivotedQr,
    };
    let mid_fit = build_spinor_thc(&[q0.clone(), q15.clone()], &grid, &spec_mid).unwrap();
    let mid = 2;
    let exact_mid = default_pair(&q15, &grid, mid, 0);
    let got_mid = selected_pair(&mid_fit, 1, mid, column);
    assert!(
        (got_mid - exact_mid).norm() < 1.0e-8,
        "{got_mid} vs {exact_mid}"
    );
    let without_qq_mid = independent_pair(
        &q15,
        &grid,
        mid,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::Skip,
        1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    let cq_mid = independent_pair(
        &q15,
        &grid,
        mid,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::CQ,
        1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    let mapped = q15
        .k_minus_q
        .iter()
        .find(|mapped| mapped.k_index == 0)
        .unwrap();
    let small_right = independent_sample(
        &q15,
        &grid,
        mid,
        mapped.k_index,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    let small_wrong = independent_sample(
        &q15,
        &grid,
        mid,
        mapped.k_index,
        0,
        false,
        false,
        SmallAngular::SameKappa,
        SmallRadial::PhysicalQ,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    let omega_field_diff = (0..2)
        .map(|spin| (small_right.small[spin] - small_wrong.small[spin]).norm())
        .fold(0.0, f64::max);
    assert!(
        (without_qq_mid - exact_mid).norm() > 1.0e-10,
        "mid-shell QQ must contribute: {without_qq_mid} vs {exact_mid}"
    );
    assert!(
        omega_field_diff > 1.0e-10,
        "Omega_-kappa and Omega_kappa small Pauli fields must differ at the mid-shell point: {omega_field_diff}"
    );
    assert!(
        (cq_mid - exact_mid).norm() > 1.0e-10,
        "mid-shell cQ scaling must not be used: {cq_mid} vs {exact_mid}"
    );

    let spec_i = SpinorThcSpec {
        rank: RankPolicy::Exact { n_mu: 1 },
        candidates: ThcCandidates::Indices(vec![3]),
        engine: ThcEngine::FullColumnPivotedQr,
    };
    let interstitial_fit = build_spinor_thc(&[q0.clone(), q15.clone()], &grid, &spec_i).unwrap();
    let i_col = q15.pair_columns.encode(0, 0, 0);
    let exact_i = default_pair(&q15, &grid, 3, 0);
    let got_i = selected_pair(&interstitial_fit, 1, 3, i_col);
    assert!((got_i - exact_i).norm() < 1.0e-8, "{got_i} vs {exact_i}");
    let wrap_flip = independent_pair(
        &q15,
        &grid,
        3,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        -1.0,
        false,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    let umklapp = independent_pair(
        &q15,
        &grid,
        3,
        0,
        0,
        0,
        false,
        false,
        SmallAngular::OppositeKappa,
        SmallRadial::PhysicalQ,
        1.0,
        true,
        MtKPhase::CellPeriodic,
        None,
        false,
        false,
    );
    assert!((got_i - wrap_flip).norm() > 1.0e-8);
    assert!((got_i - umklapp).norm() > 1.0e-8);
}

#[test]
fn build_spinor_thc_rejects_forged_finite_q_wrap_and_canonical_q() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let q0 = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let q15 = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [1.5, 0.0, 0.0])
        .unwrap();
    let grid = parent_grid(&q15);
    let spec = spec_all(1);

    let mut forged_wrap = q15.clone();
    forged_wrap.k_minus_q[0].umklapp.index[0] += 1;
    match build_spinor_thc(&[q0.clone(), forged_wrap], &grid, &spec) {
        Err(SpinorThcError::KMinusQWrap {
            q_index: 1,
            k_index: 0,
        }) => {}
        other => panic!("expected forged wrap rejection, got {other:?}"),
    }

    let mut forged_q = q15.clone();
    forged_q.source.q.cartesian[0] = InverseBohr(forged_q.source.q.cartesian[0].get() + 1.0);
    forged_q.source.interstitial_pair_support.q = forged_q.source.q;
    match build_spinor_thc(&[q0, forged_q], &grid, &spec) {
        Err(SpinorThcError::CanonicalQMismatch { q_index: 1 }) => {}
        other => panic!("expected forged canonical q rejection, got {other:?}"),
    }
}

#[test]
fn multi_q_engines_keep_zero_weight_zeta_row() {
    let physics = CheckpointPhysics::new(&hydrogen_spinor_checkpoint()).unwrap();
    let q0 = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let qh = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [0.5, 0.0, 0.0])
        .unwrap();
    let grid = parent_grid(&q0);
    let inputs = [q0.clone(), qh.clone()];
    let engines = [
        ThcEngine::FullColumnPivotedQr,
        ThcEngine::FullPivotedCholesky,
    ];
    let n_col = q0.pair_columns.n_columns().unwrap();
    let n_pts = grid.points().len();
    let mut exact = vec![Complex64::default(); n_pts * n_col];
    for p in 0..n_pts {
        for k in 0..q0.pair_columns.n_k {
            for i in 0..q0.pair_columns.n_orb {
                for j in 0..q0.pair_columns.n_orb {
                    let column = q0.pair_columns.encode(k, i, j);
                    exact[p * n_col + column] = default_pair(&qh, &grid, p, k);
                }
            }
        }
    }
    let weights: Vec<f64> = grid.points().iter().map(|point| point.weight).collect();
    let mut rank_one = Vec::new();
    let mut rank_two = Vec::new();
    for engine in engines {
        let one = build_spinor_thc(&inputs, &grid, &spec_all_engine(1, engine)).unwrap();
        let two = build_spinor_thc(&inputs, &grid, &spec_all_engine(2, engine)).unwrap();
        assert_eq!(two.records.len(), 2);
        assert_eq!(two.records[0].q_index, 0);
        assert_eq!(two.records[1].q_index, 1);
        assert_eq!(two.effective_rank, 2);
        assert_eq!(two.requested_rank, RankPolicy::Exact { n_mu: 2 });
        assert_eq!(two.records[0].layout, q0.pair_columns);
        assert_eq!(two.records[0].vertices.len(), n_col);
        assert_eq!(two.selection.provenance.strategy, SelectorStrategy::AllQL2);
        assert_eq!(two.selection.provenance.engine, L2Engine::from(engine));
        assert_eq!(
            two.selection.provenance.grid_path,
            GridPath::External {
                n_points: n_pts,
                n_candidates: 5,
            }
        );
        assert!(!two.selection.points.iter().any(|point| point.id == 1));
        assert_eq!(two.records[1].fit.n_points, n_pts);
        assert_eq!(two.records[1].fit.zeta.len(), n_pts * 2);
        rank_one.push(one);
        rank_two.push(two);
    }
    let residuals_one: Vec<f64> = rank_one
        .iter()
        .map(|result| reconstruction_residual(result, &exact, &weights, n_col, 1))
        .collect();
    let residuals_two: Vec<f64> = rank_two
        .iter()
        .map(|result| reconstruction_residual(result, &exact, &weights, n_col, 1))
        .collect();
    let exactness_floor = 1.0e-12;
    for (engine, (&rank1, &rank2)) in engines
        .iter()
        .zip(residuals_one.iter().zip(residuals_two.iter()))
    {
        assert!(rank1.is_finite() && rank1 >= 0.0, "engine={engine:?}");
        assert!(rank2.is_finite() && rank2 >= 0.0, "engine={engine:?}");
        assert!(
            rank2 < rank1.max(exactness_floor),
            "rank-2 must improve on rank-1 or stay below the exactness floor for {engine:?}: {rank2} vs {rank1}"
        );
    }
}

fn reconstruction_residual(
    result: &muffintin::SpinorThcResult,
    exact: &[Complex64],
    weights: &[f64],
    n_col: usize,
    q: usize,
) -> f64 {
    let n_pts = result.records[q].fit.n_points;
    let n_mu = result.records[q].fit.n_mu;
    let ids = result
        .selection
        .points
        .iter()
        .map(|point| point.id)
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    for &id in &ids {
        selected.extend_from_slice(&exact[id * n_col..(id + 1) * n_col]);
    }
    let mut recon = vec![Complex64::default(); n_pts * n_col];
    for p in 0..n_pts {
        for col in 0..n_col {
            let mut acc = Complex64::default();
            for mu in 0..n_mu {
                acc += result.records[q].fit.zeta[p * n_mu + mu] * selected[mu * n_col + col];
            }
            recon[p * n_col + col] = acc;
        }
    }
    let mut acc = 0.0;
    for p in 0..weights.len() {
        for col in 0..n_col {
            acc += weights[p] * (exact[p * n_col + col] - recon[p * n_col + col]).norm_sqr();
        }
    }
    acc.sqrt()
}
