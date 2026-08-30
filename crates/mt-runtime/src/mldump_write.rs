//! Shared scalar/spinor MLDUMP v1 writer helpers.

use std::f64::consts::PI;

use muffintin_prodbasis::{
    AuxiliaryPartition, CompiledAuxiliaryBasis, DiracSiteRadialSet, OrbitalPair, PartitionSite,
    SiteRadialSet, TransferQ,
};
use muffintin_core::{ExponentialMesh, ReciprocalLattice};
use muffintin_coulomb::CoulombOperator;
use muffintin_core::Cell;
use muffintin_io::{
    IoError, MLDUMP_INTERSTITIAL_SENTINEL, MLDUMP_PARENT_REGION_INTERSTITIAL,
    MLDUMP_PARENT_REGION_MUFFIN_TIN, MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY, MLDUMP_THC_ENGINE_QRCP,
    MLDUMP_THC_STRATEGY_ALL_QL2, MldumpCoulombBeginV1, MldumpCoulombGammaRefV1,
    MldumpCoulombQRecordRefV1, MldumpHeaderV1, MldumpThcBeginV1, MldumpThcParentGridRefV1,
    MldumpThcQRecordRefV1, MldumpThcSelectionRefV1, MldumpThcVertexTableRefV1,
    ScalarMldumpStreamV1, SpinorMldumpStreamV1, ValidationError,
};
use muffintin_operators::lapw::Provenance;
use muffintin_tensor::DenseEigenvectors;
use muffintin_prodbasis::thc::{L2Engine, RankPolicy, Selection, SelectorStrategy};
use num_complex::Complex64;

use crate::mldump_header::{
    HeaderBind, HeaderBindError, HeaderBindKMinusQ, HeaderBindQ, HeaderBindSite,
    preflight_mldump_header,
};
use crate::scalar_coulomb::{ScalarCoulombError, ScalarCoulombQRecord, ScalarCoulombResult};
use crate::scalar_mldump::ScalarMldumpError;
use crate::scalar_product::{ScalarKMinusQ, ScalarProductInput};
use crate::scalar_thc::ScalarThcResult;
use crate::spinor_coulomb::{SpinorCoulombError, SpinorCoulombQRecord, SpinorCoulombResult};
use crate::spinor_mldump::SpinorMldumpError;
use crate::spinor_product::{SpinorKMinusQ, SpinorProductInput};
use crate::spinor_thc::SpinorThcResult;
use crate::thc_grid::{ThcParentGrid, ThcQRecord, ThcRegion};

pub(crate) fn index_i64(value: usize) -> i64 {
    i64::try_from(value).expect("index fits i64")
}

pub(crate) trait MldumpWriteError: From<IoError> + From<ValidationError> {
    fn unsupported_engine(engine: L2Engine) -> Self;
    fn unsupported_strategy() -> Self;
    fn vertex_identity(index: usize, column: usize) -> Self;
}

impl MldumpWriteError for ScalarMldumpError {
    fn unsupported_engine(engine: L2Engine) -> Self {
        Self::UnsupportedEngine(engine)
    }

    fn unsupported_strategy() -> Self {
        Self::UnsupportedStrategy
    }

    fn vertex_identity(index: usize, column: usize) -> Self {
        Self::Coulomb(ScalarCoulombError::VertexIdentity { index, column })
    }
}

impl MldumpWriteError for SpinorMldumpError {
    fn unsupported_engine(engine: L2Engine) -> Self {
        Self::UnsupportedEngine(engine)
    }

    fn unsupported_strategy() -> Self {
        Self::UnsupportedStrategy
    }

    fn vertex_identity(index: usize, column: usize) -> Self {
        Self::Coulomb(SpinorCoulombError::VertexIdentity { index, column })
    }
}

pub(crate) trait MldumpResponseWriter {
    fn open_thc(&mut self, begin: &MldumpThcBeginV1<'_>) -> Result<(), IoError>;
    fn put_thc_q(&mut self, record: &MldumpThcQRecordRefV1<'_>) -> Result<(), IoError>;
    fn close_thc(&mut self) -> Result<(), IoError>;
    fn open_coulomb(&mut self, begin: &MldumpCoulombBeginV1) -> Result<(), IoError>;
    fn put_coulomb_q(&mut self, record: &MldumpCoulombQRecordRefV1<'_>) -> Result<(), IoError>;
    fn close_coulomb(&mut self) -> Result<(), IoError>;
}

macro_rules! impl_response_writer {
    ($ty:ty) => {
        impl MldumpResponseWriter for $ty {
            fn open_thc(&mut self, begin: &MldumpThcBeginV1<'_>) -> Result<(), IoError> {
                self.begin_thc(begin)
            }
            fn put_thc_q(&mut self, record: &MldumpThcQRecordRefV1<'_>) -> Result<(), IoError> {
                self.write_thc_q(record)
            }
            fn close_thc(&mut self) -> Result<(), IoError> {
                self.finish_thc()
            }
            fn open_coulomb(&mut self, begin: &MldumpCoulombBeginV1) -> Result<(), IoError> {
                self.begin_coulomb(begin)
            }
            fn put_coulomb_q(
                &mut self,
                record: &MldumpCoulombQRecordRefV1<'_>,
            ) -> Result<(), IoError> {
                self.write_coulomb_q(record)
            }
            fn close_coulomb(&mut self) -> Result<(), IoError> {
                self.finish_coulomb()
            }
        }
    };
}

impl_response_writer!(ScalarMldumpStreamV1);
impl_response_writer!(SpinorMldumpStreamV1);

pub(crate) trait HeaderBindInput {
    type Radial: RadialMeshSource;
    type KMinusQ: KMinusQBind;
    fn partition(&self) -> &AuxiliaryPartition;
    fn radials(&self) -> &[Self::Radial];
    fn k_fractional(&self) -> &[[f64; 3]];
    fn reciprocal(&self) -> &ReciprocalLattice;
    fn k_minus_q(&self) -> &[Self::KMinusQ];
    fn q(&self) -> &TransferQ;
}

pub(crate) trait RadialMeshSource {
    fn mesh(&self) -> &ExponentialMesh;
}

pub(crate) trait KMinusQBind {
    fn bind(&self) -> HeaderBindKMinusQ;
}

impl HeaderBindInput for ScalarProductInput {
    type Radial = SiteRadialSet;
    type KMinusQ = ScalarKMinusQ;

    fn partition(&self) -> &AuxiliaryPartition {
        &self.source.partition
    }

    fn radials(&self) -> &[Self::Radial] {
        &self.source.radials
    }

    fn k_fractional(&self) -> &[[f64; 3]] {
        &self.orbitals.k_fractional
    }

    fn reciprocal(&self) -> &ReciprocalLattice {
        &self.reciprocal
    }

    fn k_minus_q(&self) -> &[Self::KMinusQ] {
        &self.k_minus_q
    }

    fn q(&self) -> &TransferQ {
        &self.source.q
    }
}

impl HeaderBindInput for SpinorProductInput {
    type Radial = DiracSiteRadialSet;
    type KMinusQ = SpinorKMinusQ;

    fn partition(&self) -> &AuxiliaryPartition {
        &self.source.partition
    }

    fn radials(&self) -> &[Self::Radial] {
        &self.source.radials
    }

    fn k_fractional(&self) -> &[[f64; 3]] {
        &self.orbitals.k_fractional
    }

    fn reciprocal(&self) -> &ReciprocalLattice {
        &self.reciprocal
    }

    fn k_minus_q(&self) -> &[Self::KMinusQ] {
        &self.k_minus_q
    }

    fn q(&self) -> &TransferQ {
        &self.source.q
    }
}

impl RadialMeshSource for SiteRadialSet {
    fn mesh(&self) -> &ExponentialMesh {
        &self.mesh
    }
}

impl RadialMeshSource for DiracSiteRadialSet {
    fn mesh(&self) -> &ExponentialMesh {
        &self.mesh
    }
}

impl KMinusQBind for ScalarKMinusQ {
    fn bind(&self) -> HeaderBindKMinusQ {
        HeaderBindKMinusQ {
            k_index: self.k_index,
            mapped_index: self.kq_index,
            g_wrap: self.umklapp.index,
        }
    }
}

impl KMinusQBind for SpinorKMinusQ {
    fn bind(&self) -> HeaderBindKMinusQ {
        HeaderBindKMinusQ {
            k_index: self.k_index,
            mapped_index: self.kq_index,
            g_wrap: self.umklapp.index,
        }
    }
}

pub(crate) fn preflight_header<E, I>(
    header: &MldumpHeaderV1,
    first: &I,
    inputs: &[I],
    cell: &Cell,
) -> Result<(), E>
where
    E: From<HeaderBindError>,
    I: HeaderBindInput,
{
    let sites = first
        .partition()
        .sites()
        .iter()
        .zip(first.radials())
        .map(|(site, radials)| bind_site(site, radials.mesh()))
        .collect::<Vec<_>>();
    let k_maps = inputs
        .iter()
        .map(|input| {
            input
                .k_minus_q()
                .iter()
                .map(KMinusQBind::bind)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let q_records = inputs
        .iter()
        .zip(&k_maps)
        .map(|(input, k_minus_q)| HeaderBindQ {
            cartesian: input.q().cartesian.map(|component| component.get()),
            umklapp: input.q().umklapp.index,
            k_minus_q,
        })
        .collect::<Vec<_>>();
    preflight_mldump_header(
        header,
        &HeaderBind {
            direct_basis: std::array::from_fn(|row| {
                std::array::from_fn(|axis| cell.basis()[row][axis].get())
            }),
            reciprocal_basis: std::array::from_fn(|row| {
                std::array::from_fn(|axis| first.reciprocal().basis()[row][axis].get())
            }),
            cell_volume: cell.volume().get(),
            partition_volume: first.partition().interstitial().cell_volume().get(),
            sites: &sites,
            k_fractional: first.k_fractional(),
            q_records: &q_records,
        },
    )?;
    Ok(())
}

fn bind_site(site: &PartitionSite, mesh: &ExponentialMesh) -> HeaderBindSite {
    HeaderBindSite {
        position: site.position.map(|component| component.get()),
        radius: site.radius.get(),
        mesh_first: mesh.first().get(),
        mesh_increment: mesh.increment(),
        mesh_count: mesh.len(),
    }
}

pub(crate) trait ThcWriteSource {
    fn grid(&self) -> &ThcParentGrid;
    fn selection(&self) -> &Selection;
    fn requested_rank(&self) -> RankPolicy;
    fn effective_rank(&self) -> usize;
    fn records(&self) -> &[ThcQRecord];
}

impl ThcWriteSource for ScalarThcResult {
    fn grid(&self) -> &ThcParentGrid {
        &self.grid
    }

    fn selection(&self) -> &Selection {
        &self.selection
    }

    fn requested_rank(&self) -> RankPolicy {
        self.requested_rank
    }

    fn effective_rank(&self) -> usize {
        self.effective_rank
    }

    fn records(&self) -> &[ThcQRecord] {
        &self.records
    }
}

impl ThcWriteSource for SpinorThcResult {
    fn grid(&self) -> &ThcParentGrid {
        &self.grid
    }

    fn selection(&self) -> &Selection {
        &self.selection
    }

    fn requested_rank(&self) -> RankPolicy {
        self.requested_rank
    }

    fn effective_rank(&self) -> usize {
        self.effective_rank
    }

    fn records(&self) -> &[ThcQRecord] {
        &self.records
    }
}

pub(crate) fn write_thc<S, E, T>(stream: &mut S, thc: &T) -> Result<(), E>
where
    S: MldumpResponseWriter,
    E: MldumpWriteError,
    T: ThcWriteSource,
{
    if thc.selection().provenance.strategy != SelectorStrategy::AllQL2 {
        return Err(E::unsupported_strategy());
    }
    let engine = match thc.selection().provenance.engine {
        L2Engine::FullColumnPivotedQr => MLDUMP_THC_ENGINE_QRCP,
        L2Engine::FullPivotedCholesky => MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY,
        other => return Err(E::unsupported_engine(other)),
    };
    let n_points = thc.grid().points().len();
    let mut coordinates = Vec::with_capacity(n_points * 3);
    let mut weights = Vec::with_capacity(n_points);
    let mut region_kind = Vec::with_capacity(n_points);
    let mut site_index = Vec::with_capacity(n_points);
    let mut radial_index = Vec::with_capacity(n_points);
    for point in thc.grid().points() {
        coordinates.extend(point.coordinate.iter().map(|component| component.get()));
        weights.push(point.weight);
        match point.region {
            ThcRegion::MuffinTin {
                site,
                radial_index: radial,
            } => {
                region_kind.push(MLDUMP_PARENT_REGION_MUFFIN_TIN);
                site_index.push(index_i64(site));
                radial_index.push(index_i64(radial));
            }
            ThcRegion::Interstitial => {
                region_kind.push(MLDUMP_PARENT_REGION_INTERSTITIAL);
                site_index.push(MLDUMP_INTERSTITIAL_SENTINEL);
                radial_index.push(MLDUMP_INTERSTITIAL_SENTINEL);
            }
        }
    }
    let pivots = thc
        .selection()
        .pivots
        .iter()
        .map(|index| index_i64(*index))
        .collect::<Vec<_>>();
    let points = thc
        .selection()
        .points
        .iter()
        .map(|point| index_i64(point.id))
        .collect::<Vec<_>>();
    let n_candidates = weights.iter().filter(|weight| **weight > 0.0).count();
    let requested_rank = match thc.requested_rank() {
        RankPolicy::Exact { n_mu } => n_mu,
        RankPolicy::Threshold { n_max, .. } => n_max,
    };
    let grid_provenance = provenance_key(thc.grid().provenance());
    stream.open_thc(&MldumpThcBeginV1 {
        parent_grid: MldumpThcParentGridRefV1 {
            n_points,
            coordinates: &coordinates,
            weights: &weights,
            region_kind: &region_kind,
            site_index: &site_index,
            radial_index: &radial_index,
            provenance: &grid_provenance,
        },
        strategy: MLDUMP_THC_STRATEGY_ALL_QL2,
        engine,
        requested_rank,
        effective_rank: thc.effective_rank(),
        n_candidates,
        selection: MldumpThcSelectionRefV1 {
            pivots: &pivots,
            points: &points,
        },
    })?;
    for record in thc.records() {
        let zeta = flatten_complex(&record.fit.zeta);
        let n_vertex = record.vertices.len();
        let mut column = Vec::with_capacity(n_vertex);
        let mut k_left_right = Vec::with_capacity(n_vertex * 3);
        let mut coefficients = Vec::new();
        for (index, vertex) in record.vertices.iter().enumerate() {
            column.push(index_i64(index));
            match vertex.pair() {
                OrbitalPair::Bloch {
                    k_index,
                    left,
                    right,
                } => {
                    k_left_right.push(index_i64(k_index));
                    k_left_right.push(index_i64(left));
                    k_left_right.push(index_i64(right));
                }
                _ => return Err(E::vertex_identity(record.q_index, index)),
            }
            coefficients.extend(flatten_complex(vertex.coefficients()));
        }
        let layout_provenance = provenance_key(&record.auxiliary.provenance);
        stream.put_thc_q(&MldumpThcQRecordRefV1 {
            q_index: record.q_index,
            aux_dimension: record.fit.n_mu,
            layout_provenance: &layout_provenance,
            zeta: &zeta,
            residual_l2_all_frobenius: record.fit.l2_all.frobenius,
            residual_l2_all_column_max: record.fit.l2_all.column_max,
            vertices: MldumpThcVertexTableRefV1 {
                n_vertex,
                column: &column,
                k_left_right: &k_left_right,
                coefficients: &coefficients,
            },
        })?;
    }
    stream.close_thc()?;
    Ok(())
}

pub(crate) trait CoulombQWriteSource {
    fn q_index(&self) -> usize;
    fn operator(&self) -> &CoulombOperator;
    fn auxiliary(&self) -> &CompiledAuxiliaryBasis;
}

impl CoulombQWriteSource for ScalarCoulombQRecord {
    fn q_index(&self) -> usize {
        self.q_index
    }

    fn operator(&self) -> &CoulombOperator {
        &self.operator
    }

    fn auxiliary(&self) -> &CompiledAuxiliaryBasis {
        &self.auxiliary
    }
}

impl CoulombQWriteSource for SpinorCoulombQRecord {
    fn q_index(&self) -> usize {
        self.q_index
    }

    fn operator(&self) -> &CoulombOperator {
        &self.operator
    }

    fn auxiliary(&self) -> &CompiledAuxiliaryBasis {
        &self.auxiliary
    }
}

pub(crate) fn write_coulomb<S, E, R>(
    stream: &mut S,
    lexp: u32,
    interpolation_l_max: u32,
    interpolation_pw_cutoff: f64,
    records: &[R],
) -> Result<(), E>
where
    S: MldumpResponseWriter,
    E: From<IoError>,
    R: CoulombQWriteSource,
{
    stream.open_coulomb(&MldumpCoulombBeginV1 {
        lexp,
        interpolation_l_max,
        interpolation_pw_cutoff,
    })?;
    for record in records {
        let body = flatten_complex(record.operator().matrix());
        let gamma_scratch = record.operator().gamma().map(|gamma| {
            (
                gamma.spherical_average_subtracted,
                gamma.head_prefactor,
                flatten_complex(&gamma.constant_coefficients),
            )
        });
        let layout_provenance = provenance_key(&record.auxiliary().provenance);
        stream.put_coulomb_q(&MldumpCoulombQRecordRefV1 {
            q_index: record.q_index(),
            aux_dimension: record.operator().dimension(),
            layout_provenance: &layout_provenance,
            body: &body,
            gamma: gamma_scratch
                .as_ref()
                .map(|(subtracted, prefactor, coeffs)| MldumpCoulombGammaRefV1 {
                    spherical_average_subtracted: *subtracted,
                    head_prefactor: *prefactor,
                    constant_coefficients: coeffs,
                }),
        })?;
    }
    stream.close_coulomb()?;
    Ok(())
}

pub(crate) fn write_coulomb_result<S, E>(
    stream: &mut S,
    coulomb: &ScalarCoulombResult,
) -> Result<(), E>
where
    S: MldumpResponseWriter,
    E: From<IoError>,
{
    write_coulomb(
        stream,
        coulomb.context.request.lexp(),
        coulomb.context.projection.l_max,
        coulomb.context.projection.pw_cutoff.get(),
        &coulomb.records,
    )
}

pub(crate) fn write_spinor_coulomb_result<S, E>(
    stream: &mut S,
    coulomb: &SpinorCoulombResult,
) -> Result<(), E>
where
    S: MldumpResponseWriter,
    E: From<IoError>,
{
    write_coulomb(
        stream,
        coulomb.context.request.lexp(),
        coulomb.context.projection.l_max,
        coulomb.context.projection.pw_cutoff.get(),
        &coulomb.records,
    )
}

pub(crate) fn flatten_eigenvectors(
    evecs: &DenseEigenvectors,
    n_orb: usize,
) -> Result<Vec<f64>, ValidationError> {
    let n_basis = evecs.rows();
    let mut out = Vec::with_capacity(n_basis * n_orb * 2);
    for row in 0..n_basis {
        for band in 0..n_orb {
            let value = evecs
                .get(row, band)
                .map_err(|error| ValidationError::InvalidValue {
                    path: "orbitals.eigenvectors".to_owned(),
                    expected: "in-bounds leading window".to_owned(),
                    actual: error.to_string(),
                })?;
            out.push(value.re);
            out.push(value.im);
        }
    }
    Ok(out)
}

pub(crate) fn flatten_complex(values: &[Complex64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.push(value.re);
        out.push(value.im);
    }
    out
}

pub(crate) fn provenance_key(provenance: &Provenance) -> String {
    format!(
        "{}|{}",
        provenance.recipe.as_deref().unwrap_or(""),
        provenance.reference.as_deref().unwrap_or("")
    )
}

pub(crate) fn interstitial_volume(partition: &AuxiliaryPartition) -> f64 {
    let cell = partition.interstitial().cell_volume().get();
    let muffin = partition
        .sites()
        .iter()
        .map(|site| 4.0 / 3.0 * PI * site.radius.get().powi(3))
        .sum::<f64>();
    cell - muffin
}
