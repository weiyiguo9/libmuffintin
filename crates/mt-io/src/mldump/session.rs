//! Shared MLDUMP v1 streaming session. Scalar and spinor lanes share the
//! products/THC/Coulomb state machine; orbitals stay lane-specific.

use hdf5_metno::File;

use super::response::{
    CoulombAlignmentSummary, MldumpCoulombBeginV1, MldumpCoulombQRecordRefV1, MldumpThcBeginV1,
    MldumpThcQRecordRefV1, OrbitalAlignmentSummary, ProductAlignmentSummary, ThcAlignmentSummary,
};
use super::scalar_orbitals::{
    MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON, ScalarOrbitalKRefV1, ScalarOrbitalsBeginV1,
};
use super::scalar_products::{
    ScalarProductQRecordRefV1, ScalarProductSiteRefV1, ScalarProductsBeginV1,
};
use super::spinor_orbitals::{
    MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, SpinorOrbitalKRefV1, SpinorOrbitalsBeginV1,
};
use super::spinor_products::{
    SpinorProductQRecordRefV1, SpinorProductSiteRefV1, SpinorProductsBeginV1,
};
use super::{MldumpHeaderV1, require_record_capacity, section_already_written, stream_state_error};
use crate::error::{IoError, ValidationError};

/// Section-specific HDF5 payload writers for one representation lane.
pub(crate) trait MldumpLane: Sized {
    /// Cursor stored while that lane's `/orbitals` section is open.
    type OrbitalsCursor: Copy + Eq + std::fmt::Debug;
    /// Small retained orbital counters/summaries.
    type OrbitalsState: std::fmt::Debug;
    /// Shared `/products` begin record.
    type ProductsBegin<'a>;
    /// One site radial record.
    type ProductSite<'a>;
    /// One product $q$ record.
    type ProductQ<'a>;

    /// Path token used in session-state errors (`scalar` / `spinor`).
    const KIND: &'static str;
    /// `/orbitals/@representation` and companion THC/Coulomb tags.
    const REPRESENTATION: &'static str;

    fn new_orbitals_state(header: &MldumpHeaderV1) -> Self::OrbitalsState;

    fn begin_products(
        file: &File,
        header: &MldumpHeaderV1,
        begin: &Self::ProductsBegin<'_>,
    ) -> Result<(), IoError>;

    fn products_dims(begin: &Self::ProductsBegin<'_>) -> (usize, usize);

    fn write_product_site(
        file: &File,
        header: &MldumpHeaderV1,
        site: usize,
        record: &Self::ProductSite<'_>,
    ) -> Result<(), IoError>;

    fn product_site_index(record: &Self::ProductSite<'_>) -> usize;

    fn write_product_q(file: &File, q: usize, record: &Self::ProductQ<'_>) -> Result<(), IoError>;

    fn product_q_index(record: &Self::ProductQ<'_>) -> usize;

    fn product_q_binding(record: &Self::ProductQ<'_>) -> (usize, [f64; 3], [i32; 3]);

    fn finish_sections(
        header: &MldumpHeaderV1,
        orbitals: &Self::OrbitalsState,
        products: Option<&ProductAlignmentSummary>,
        thc: Option<&ThcAlignmentSummary>,
        coulomb: Option<&CoulombAlignmentSummary>,
    ) -> Result<(), IoError>;
}

/// Scalar Koelling–Harmon streaming lane.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ScalarLane;

/// Full first-variation spinor streaming lane.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SpinorLane;

/// Streaming payload session parameterized by representation lane.
#[derive(Debug)]
pub(crate) struct MldumpStreamV1<L: MldumpLane> {
    file: File,
    header: MldumpHeaderV1,
    phase: StreamPhase<L::OrbitalsCursor>,
    orbitals: L::OrbitalsState,
    product_summary: Option<ProductAlignmentSummary>,
    thc_summary: Option<ThcAlignmentSummary>,
    coulomb_summary: Option<CoulombAlignmentSummary>,
    products_n_site: usize,
    thc_n_parent: usize,
    thc_effective_rank: usize,
}

/// Streaming scalar payload session.
#[derive(Debug)]
pub struct ScalarMldumpStreamV1 {
    inner: MldumpStreamV1<ScalarLane>,
}

/// Streaming spinor payload session.
#[derive(Debug)]
pub struct SpinorMldumpStreamV1 {
    inner: MldumpStreamV1<SpinorLane>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamPhase<O> {
    Start,
    Orbitals(O),
    Products { next_site: usize, next_q: usize },
    Thc { next_q: usize },
    Coulomb { next_q: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ScalarOrbitalsCursor {
    next_spin: usize,
    next_k: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SpinorOrbitalsCursor {
    next_k: usize,
}

#[derive(Debug)]
pub(crate) struct ScalarOrbitalsState {
    summary: Option<OrbitalAlignmentSummary>,
    band_window: usize,
    spin_count: usize,
}

#[derive(Debug)]
pub(crate) struct SpinorOrbitalsState {
    written: bool,
    band_window: usize,
    n_k: usize,
}

impl MldumpLane for ScalarLane {
    type OrbitalsCursor = ScalarOrbitalsCursor;
    type OrbitalsState = ScalarOrbitalsState;
    type ProductsBegin<'a> = ScalarProductsBeginV1<'a>;
    type ProductSite<'a> = ScalarProductSiteRefV1<'a>;
    type ProductQ<'a> = ScalarProductQRecordRefV1<'a>;

    const KIND: &'static str = "scalar";
    const REPRESENTATION: &'static str = MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON;

    fn new_orbitals_state(_header: &MldumpHeaderV1) -> Self::OrbitalsState {
        ScalarOrbitalsState {
            summary: None,
            band_window: 0,
            spin_count: 0,
        }
    }

    fn begin_products(
        file: &File,
        header: &MldumpHeaderV1,
        begin: &Self::ProductsBegin<'_>,
    ) -> Result<(), IoError> {
        super::scalar_products::begin_scalar_products(file, header, begin)
    }

    fn products_dims(begin: &Self::ProductsBegin<'_>) -> (usize, usize) {
        (begin.n_k, begin.n_orb)
    }

    fn write_product_site(
        file: &File,
        header: &MldumpHeaderV1,
        site: usize,
        record: &Self::ProductSite<'_>,
    ) -> Result<(), IoError> {
        super::scalar_products::write_scalar_product_site(file, header, site, record)
    }

    fn product_site_index(record: &Self::ProductSite<'_>) -> usize {
        record.site_index
    }

    fn write_product_q(file: &File, q: usize, record: &Self::ProductQ<'_>) -> Result<(), IoError> {
        super::scalar_products::write_scalar_product_q(file, q, record)
    }

    fn product_q_index(record: &Self::ProductQ<'_>) -> usize {
        record.q_index
    }

    fn product_q_binding(record: &Self::ProductQ<'_>) -> (usize, [f64; 3], [i32; 3]) {
        (
            record.q_index,
            record.transfer_cartesian,
            record.global_transfer,
        )
    }

    fn finish_sections(
        header: &MldumpHeaderV1,
        orbitals: &Self::OrbitalsState,
        products: Option<&ProductAlignmentSummary>,
        thc: Option<&ThcAlignmentSummary>,
        coulomb: Option<&CoulombAlignmentSummary>,
    ) -> Result<(), IoError> {
        match (orbitals.summary.as_ref(), products, thc, coulomb) {
            (Some(orbitals), Some(products), Some(thc), Some(coulomb)) => {
                super::response::validate_scalar_alignment(header, orbitals, products, thc, coulomb)
            }
            _ => Err(missing_section_summary(
                Self::KIND,
                "missing retained summary",
            )),
        }
    }
}

impl MldumpLane for SpinorLane {
    type OrbitalsCursor = SpinorOrbitalsCursor;
    type OrbitalsState = SpinorOrbitalsState;
    type ProductsBegin<'a> = SpinorProductsBeginV1<'a>;
    type ProductSite<'a> = SpinorProductSiteRefV1<'a>;
    type ProductQ<'a> = SpinorProductQRecordRefV1<'a>;

    const KIND: &'static str = "spinor";
    const REPRESENTATION: &'static str = MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION;

    fn new_orbitals_state(header: &MldumpHeaderV1) -> Self::OrbitalsState {
        SpinorOrbitalsState {
            written: false,
            band_window: 0,
            n_k: header.mesh.k_points.len(),
        }
    }

    fn begin_products(
        file: &File,
        header: &MldumpHeaderV1,
        begin: &Self::ProductsBegin<'_>,
    ) -> Result<(), IoError> {
        super::spinor_products::begin_spinor_products(file, header, begin)
    }

    fn products_dims(begin: &Self::ProductsBegin<'_>) -> (usize, usize) {
        (begin.n_k, begin.n_orb)
    }

    fn write_product_site(
        file: &File,
        header: &MldumpHeaderV1,
        site: usize,
        record: &Self::ProductSite<'_>,
    ) -> Result<(), IoError> {
        super::spinor_products::write_spinor_product_site(file, header, site, record)
    }

    fn product_site_index(record: &Self::ProductSite<'_>) -> usize {
        record.site_index
    }

    fn write_product_q(file: &File, q: usize, record: &Self::ProductQ<'_>) -> Result<(), IoError> {
        super::spinor_products::write_spinor_product_q(file, q, record)
    }

    fn product_q_index(record: &Self::ProductQ<'_>) -> usize {
        record.q_index
    }

    fn product_q_binding(record: &Self::ProductQ<'_>) -> (usize, [f64; 3], [i32; 3]) {
        (
            record.q_index,
            record.transfer_cartesian,
            record.global_transfer,
        )
    }

    fn finish_sections(
        header: &MldumpHeaderV1,
        orbitals: &Self::OrbitalsState,
        products: Option<&ProductAlignmentSummary>,
        thc: Option<&ThcAlignmentSummary>,
        coulomb: Option<&CoulombAlignmentSummary>,
    ) -> Result<(), IoError> {
        if !orbitals.written {
            return Err(missing_section_summary(
                Self::KIND,
                "missing orbitals summary",
            ));
        }
        match (products, thc, coulomb) {
            (Some(products), Some(thc), Some(coulomb)) => {
                super::response::validate_payload_alignment(
                    header,
                    orbitals.band_window,
                    products,
                    thc,
                    coulomb,
                )
            }
            _ => Err(missing_section_summary(
                Self::KIND,
                "missing retained summary",
            )),
        }
    }
}

impl<L: MldumpLane> MldumpStreamV1<L> {
    pub(crate) fn new(file: File, header: MldumpHeaderV1) -> Self {
        Self {
            orbitals: L::new_orbitals_state(&header),
            file,
            header,
            phase: StreamPhase::Start,
            product_summary: None,
            thc_summary: None,
            coulomb_summary: None,
            products_n_site: 0,
            thc_n_parent: 0,
            thc_effective_rank: 0,
        }
    }

    /// Open `/products` and write shared partition binding.
    pub fn begin_products(&mut self, begin: &L::ProductsBegin<'_>) -> Result<(), IoError> {
        self.require_idle("begin_products")?;
        if self.product_summary.is_some() {
            return Err(section_already_written("/products"));
        }
        L::begin_products(&self.file, &self.header, begin)?;
        self.products_n_site = self.header.geometry.sites.len();
        let (n_k, n_orb) = L::products_dims(begin);
        self.product_summary = Some(ProductAlignmentSummary::new(n_k, n_orb));
        self.phase = StreamPhase::Products {
            next_site: 0,
            next_q: 0,
        };
        Ok(())
    }

    /// Write one site radial record immediately.
    pub fn write_product_site(&mut self, record: &L::ProductSite<'_>) -> Result<(), IoError> {
        let next_site = match self.phase {
            StreamPhase::Products {
                next_site,
                next_q: 0,
            } => next_site,
            _ => {
                return Err(
                    self.phase_error("write_product_site", "product sites before q records")
                );
            }
        };
        require_record_capacity("products.sites", next_site, self.products_n_site)?;
        let site_index = L::product_site_index(record);
        if site_index != next_site {
            return Err(ValidationError::InvalidValue {
                path: "products.sites".to_owned(),
                expected: next_site.to_string(),
                actual: site_index.to_string(),
            }
            .into());
        }
        L::write_product_site(&self.file, &self.header, next_site, record)?;
        self.phase = StreamPhase::Products {
            next_site: next_site + 1,
            next_q: 0,
        };
        Ok(())
    }

    /// Write one positional product $q$ record immediately.
    pub fn write_product_q(&mut self, record: &L::ProductQ<'_>) -> Result<(), IoError> {
        let (next_site, next_q) = match self.phase {
            StreamPhase::Products { next_site, next_q } => (next_site, next_q),
            _ => {
                return Err(self.phase_error("write_product_q", "products section in progress"));
            }
        };
        if next_site != self.products_n_site {
            return Err(ValidationError::InvalidValue {
                path: "products.q_records".to_owned(),
                expected: format!("{} site records first", self.products_n_site),
                actual: format!("{next_site} sites written"),
            }
            .into());
        }
        require_record_capacity(
            "products.q_records",
            next_q,
            self.header.mesh.q_entries.len(),
        )?;
        let q_index = L::product_q_index(record);
        if q_index != next_q {
            return Err(ValidationError::InvalidValue {
                path: "products.q_records".to_owned(),
                expected: next_q.to_string(),
                actual: q_index.to_string(),
            }
            .into());
        }
        L::write_product_q(&self.file, next_q, record)?;
        if let Some(summary) = self.product_summary.as_mut() {
            let (q_index, transfer_cartesian, global_transfer) = L::product_q_binding(record);
            summary.push_q_binding(q_index, transfer_cartesian, global_transfer);
        }
        self.phase = StreamPhase::Products {
            next_site,
            next_q: next_q + 1,
        };
        Ok(())
    }

    /// Close `/products` after every site and $q$ record has been written.
    pub fn finish_products(&mut self) -> Result<(), IoError> {
        let StreamPhase::Products { next_site, next_q } = self.phase else {
            return Err(self.phase_error("finish_products", "products section in progress"));
        };
        let n_q = self.header.mesh.q_entries.len();
        if next_site != self.products_n_site || next_q != n_q {
            return Err(ValidationError::InvalidValue {
                path: "products".to_owned(),
                expected: format!("{} sites and {n_q} q records", self.products_n_site),
                actual: format!("sites={next_site} q={next_q}"),
            }
            .into());
        }
        self.phase = StreamPhase::Start;
        Ok(())
    }

    /// Open `/thc` and write the shared parent grid and selection.
    pub fn begin_thc(&mut self, begin: &MldumpThcBeginV1<'_>) -> Result<(), IoError> {
        self.require_idle("begin_thc")?;
        if self.thc_summary.is_some() {
            return Err(section_already_written("/thc"));
        }
        if self.product_summary.is_none() {
            return Err(ValidationError::InvalidValue {
                path: "/thc".to_owned(),
                expected: "products section written before thc".to_owned(),
                actual: "products summary missing".to_owned(),
            }
            .into());
        }
        super::response::begin_mldump_thc(&self.file, &self.header, begin, L::REPRESENTATION)?;
        self.thc_n_parent = begin.parent_grid.n_points;
        self.thc_effective_rank = begin.effective_rank;
        self.thc_summary = Some(ThcAlignmentSummary::new());
        self.phase = StreamPhase::Thc { next_q: 0 };
        Ok(())
    }

    /// Write one THC $q$ record immediately.
    pub fn write_thc_q(&mut self, record: &MldumpThcQRecordRefV1<'_>) -> Result<(), IoError> {
        let next_q = match self.phase {
            StreamPhase::Thc { next_q } => next_q,
            _ => return Err(self.phase_error("write_thc_q", "thc section in progress")),
        };
        require_record_capacity("thc.q_records", next_q, self.header.mesh.q_entries.len())?;
        if record.q_index != next_q {
            return Err(ValidationError::InvalidValue {
                path: "thc.q_records".to_owned(),
                expected: next_q.to_string(),
                actual: record.q_index.to_string(),
            }
            .into());
        }
        let (n_k, n_orb) = match self.product_summary.as_ref() {
            Some(products) => (products.n_k, products.n_orb),
            None => {
                return Err(ValidationError::InvalidValue {
                    path: "thc.q_records".to_owned(),
                    expected: "products section written before thc".to_owned(),
                    actual: "products summary missing".to_owned(),
                }
                .into());
            }
        };
        super::response::write_mldump_thc_q(
            &self.file,
            next_q,
            self.thc_n_parent,
            self.thc_effective_rank,
            n_k,
            n_orb,
            record,
        )?;
        if let Some(summary) = self.thc_summary.as_mut() {
            summary.push_q(record);
        }
        self.phase = StreamPhase::Thc { next_q: next_q + 1 };
        Ok(())
    }

    /// Close `/thc` after every mesh $q$ record has been written.
    pub fn finish_thc(&mut self) -> Result<(), IoError> {
        let StreamPhase::Thc { next_q } = self.phase else {
            return Err(self.phase_error("finish_thc", "thc section in progress"));
        };
        let n_q = self.header.mesh.q_entries.len();
        if next_q != n_q {
            return Err(ValidationError::InvalidValue {
                path: "thc".to_owned(),
                expected: format!("{n_q} q records"),
                actual: next_q.to_string(),
            }
            .into());
        }
        self.phase = StreamPhase::Start;
        Ok(())
    }

    /// Open `/coulomb` and write request/projection attributes.
    pub fn begin_coulomb(&mut self, begin: &MldumpCoulombBeginV1) -> Result<(), IoError> {
        self.require_idle("begin_coulomb")?;
        if self.coulomb_summary.is_some() {
            return Err(section_already_written("/coulomb"));
        }
        super::response::begin_mldump_coulomb(&self.file, begin, L::REPRESENTATION)?;
        self.coulomb_summary = Some(CoulombAlignmentSummary::new());
        self.phase = StreamPhase::Coulomb { next_q: 0 };
        Ok(())
    }

    /// Write one Coulomb $q$ record immediately.
    pub fn write_coulomb_q(
        &mut self,
        record: &MldumpCoulombQRecordRefV1<'_>,
    ) -> Result<(), IoError> {
        let next_q = match self.phase {
            StreamPhase::Coulomb { next_q } => next_q,
            _ => {
                return Err(self.phase_error("write_coulomb_q", "coulomb section in progress"));
            }
        };
        require_record_capacity(
            "coulomb.q_records",
            next_q,
            self.header.mesh.q_entries.len(),
        )?;
        if record.q_index != next_q {
            return Err(ValidationError::InvalidValue {
                path: "coulomb.q_records".to_owned(),
                expected: next_q.to_string(),
                actual: record.q_index.to_string(),
            }
            .into());
        }
        super::response::write_mldump_coulomb_q(&self.file, next_q, record)?;
        if let Some(summary) = self.coulomb_summary.as_mut() {
            summary.push_q(record);
        }
        self.phase = StreamPhase::Coulomb { next_q: next_q + 1 };
        Ok(())
    }

    /// Close `/coulomb` after every mesh $q$ record has been written.
    pub fn finish_coulomb(&mut self) -> Result<(), IoError> {
        let StreamPhase::Coulomb { next_q } = self.phase else {
            return Err(self.phase_error("finish_coulomb", "coulomb section in progress"));
        };
        let n_q = self.header.mesh.q_entries.len();
        if next_q != n_q {
            return Err(ValidationError::InvalidValue {
                path: "coulomb".to_owned(),
                expected: format!("{n_q} q records"),
                actual: next_q.to_string(),
            }
            .into());
        }
        self.phase = StreamPhase::Start;
        Ok(())
    }

    /// Finish the populated file after all four sections.
    pub fn finish(self) -> Result<(), IoError> {
        if self.phase != StreamPhase::Start {
            return Err(self.phase_error("finish", "no section left open"));
        }
        L::finish_sections(
            &self.header,
            &self.orbitals,
            self.product_summary.as_ref(),
            self.thc_summary.as_ref(),
            self.coulomb_summary.as_ref(),
        )
    }

    fn require_idle(&self, method: &str) -> Result<(), IoError> {
        if matches!(self.phase, StreamPhase::Start) {
            Ok(())
        } else {
            Err(self.phase_error(method, "no section currently open"))
        }
    }

    fn phase_error(&self, method: &str, expected: &str) -> IoError {
        stream_state_error(L::KIND, method, expected)
    }
}

impl MldumpStreamV1<ScalarLane> {
    /// Open `/orbitals` and write shared attributes. Spin/$k$ records follow.
    pub fn begin_orbitals(&mut self, begin: &ScalarOrbitalsBeginV1) -> Result<(), IoError> {
        self.require_idle("begin_orbitals")?;
        if self.orbitals.summary.is_some() {
            return Err(section_already_written("/orbitals"));
        }
        super::scalar_orbitals::begin_scalar_orbitals(&self.file, begin)?;
        self.orbitals.band_window = begin.band_window_count;
        self.orbitals.spin_count = begin.spin_count;
        self.orbitals.summary = Some(OrbitalAlignmentSummary::new(
            begin.spin_count,
            self.header.mesh.k_points.len(),
            begin.band_window_count,
        ));
        self.phase = StreamPhase::Orbitals(ScalarOrbitalsCursor {
            next_spin: 0,
            next_k: 0,
        });
        Ok(())
    }

    /// Write one spin/$k$ orbital record immediately.
    pub fn write_orbital_k(
        &mut self,
        spin: usize,
        record: &ScalarOrbitalKRefV1<'_>,
    ) -> Result<(), IoError> {
        let (next_spin, next_k) = match self.phase {
            StreamPhase::Orbitals(ScalarOrbitalsCursor { next_spin, next_k }) => {
                (next_spin, next_k)
            }
            _ => {
                return Err(self.phase_error("write_orbital_k", "orbitals section in progress"));
            }
        };
        require_record_capacity("orbitals.record", next_spin, self.orbitals.spin_count)?;
        require_record_capacity("orbitals.record", next_k, self.header.mesh.k_points.len())?;
        if spin != next_spin || record.k_index != next_k {
            return Err(ValidationError::InvalidValue {
                path: "orbitals.record".to_owned(),
                expected: format!("spin={next_spin} k={next_k}"),
                actual: format!("spin={spin} k={}", record.k_index),
            }
            .into());
        }
        super::scalar_orbitals::write_scalar_orbital_k(
            &self.file,
            &self.header,
            spin,
            self.orbitals.band_window,
            record,
        )?;
        let n_k = self.header.mesh.k_points.len();
        let (next_spin, next_k) = if next_k + 1 == n_k {
            (next_spin + 1, 0)
        } else {
            (next_spin, next_k + 1)
        };
        self.phase = StreamPhase::Orbitals(ScalarOrbitalsCursor { next_spin, next_k });
        Ok(())
    }

    /// Close `/orbitals` after every spin/$k$ record has been written.
    pub fn finish_orbitals(&mut self) -> Result<(), IoError> {
        let StreamPhase::Orbitals(ScalarOrbitalsCursor { next_spin, next_k }) = self.phase else {
            return Err(self.phase_error("finish_orbitals", "orbitals section in progress"));
        };
        if next_spin != self.orbitals.spin_count || next_k != 0 {
            return Err(ValidationError::InvalidValue {
                path: "orbitals".to_owned(),
                expected: format!(
                    "{} spins × {} k records",
                    self.orbitals.spin_count,
                    self.header.mesh.k_points.len()
                ),
                actual: format!("next spin={next_spin} k={next_k}"),
            }
            .into());
        }
        self.phase = StreamPhase::Start;
        Ok(())
    }
}

impl MldumpStreamV1<SpinorLane> {
    /// Open `/orbitals` and write shared attributes. $k$ records follow.
    pub fn begin_orbitals(&mut self, begin: &SpinorOrbitalsBeginV1) -> Result<(), IoError> {
        self.require_idle("begin_orbitals")?;
        if self.orbitals.written {
            return Err(section_already_written("/orbitals"));
        }
        super::spinor_orbitals::begin_spinor_orbitals(&self.file, begin)?;
        self.orbitals.band_window = begin.band_window_count;
        self.orbitals.written = true;
        self.phase = StreamPhase::Orbitals(SpinorOrbitalsCursor { next_k: 0 });
        Ok(())
    }

    /// Write one $k$ orbital record immediately.
    pub fn write_orbital_k(&mut self, record: &SpinorOrbitalKRefV1<'_>) -> Result<(), IoError> {
        let next_k = match self.phase {
            StreamPhase::Orbitals(SpinorOrbitalsCursor { next_k }) => next_k,
            _ => {
                return Err(self.phase_error("write_orbital_k", "orbitals section in progress"));
            }
        };
        require_record_capacity("orbitals.record", next_k, self.orbitals.n_k)?;
        if record.k_index != next_k {
            return Err(ValidationError::InvalidValue {
                path: "orbitals.record".to_owned(),
                expected: format!("k={next_k}"),
                actual: format!("k={}", record.k_index),
            }
            .into());
        }
        super::spinor_orbitals::write_spinor_orbital_k(
            &self.file,
            &self.header,
            self.orbitals.band_window,
            record,
        )?;
        self.phase = StreamPhase::Orbitals(SpinorOrbitalsCursor { next_k: next_k + 1 });
        Ok(())
    }

    /// Close `/orbitals` after every $k$ record has been written.
    pub fn finish_orbitals(&mut self) -> Result<(), IoError> {
        let StreamPhase::Orbitals(SpinorOrbitalsCursor { next_k }) = self.phase else {
            return Err(self.phase_error("finish_orbitals", "orbitals section in progress"));
        };
        if next_k != self.orbitals.n_k {
            return Err(ValidationError::InvalidValue {
                path: "orbitals".to_owned(),
                expected: format!("{} k records", self.orbitals.n_k),
                actual: format!("next k={next_k}"),
            }
            .into());
        }
        self.phase = StreamPhase::Start;
        Ok(())
    }
}

fn missing_section_summary(kind: &str, actual: &str) -> IoError {
    ValidationError::InvalidValue {
        path: kind.to_owned(),
        expected: "alignment summaries for all four written sections".to_owned(),
        actual: actual.to_owned(),
    }
    .into()
}

impl ScalarMldumpStreamV1 {
    pub(crate) fn new(file: File, header: MldumpHeaderV1) -> Self {
        Self {
            inner: MldumpStreamV1::new(file, header),
        }
    }

    /// Open `/orbitals` and write shared attributes. Spin/$k$ records follow.
    pub fn begin_orbitals(&mut self, begin: &ScalarOrbitalsBeginV1) -> Result<(), IoError> {
        self.inner.begin_orbitals(begin)
    }

    /// Write one spin/$k$ orbital record immediately.
    pub fn write_orbital_k(
        &mut self,
        spin: usize,
        record: &ScalarOrbitalKRefV1<'_>,
    ) -> Result<(), IoError> {
        self.inner.write_orbital_k(spin, record)
    }

    /// Close `/orbitals` after every spin/$k$ record has been written.
    pub fn finish_orbitals(&mut self) -> Result<(), IoError> {
        self.inner.finish_orbitals()
    }

    /// Open `/products` and write shared partition binding.
    pub fn begin_products(&mut self, begin: &ScalarProductsBeginV1<'_>) -> Result<(), IoError> {
        self.inner.begin_products(begin)
    }

    /// Write one site radial record immediately.
    pub fn write_product_site(
        &mut self,
        record: &ScalarProductSiteRefV1<'_>,
    ) -> Result<(), IoError> {
        self.inner.write_product_site(record)
    }

    /// Write one positional product $q$ record immediately.
    pub fn write_product_q(
        &mut self,
        record: &ScalarProductQRecordRefV1<'_>,
    ) -> Result<(), IoError> {
        self.inner.write_product_q(record)
    }

    /// Close `/products` after every site and $q$ record has been written.
    pub fn finish_products(&mut self) -> Result<(), IoError> {
        self.inner.finish_products()
    }

    /// Open `/thc` and write the shared parent grid and selection.
    pub fn begin_thc(&mut self, begin: &MldumpThcBeginV1<'_>) -> Result<(), IoError> {
        self.inner.begin_thc(begin)
    }

    /// Write one THC $q$ record immediately.
    pub fn write_thc_q(&mut self, record: &MldumpThcQRecordRefV1<'_>) -> Result<(), IoError> {
        self.inner.write_thc_q(record)
    }

    /// Close `/thc` after every mesh $q$ record has been written.
    pub fn finish_thc(&mut self) -> Result<(), IoError> {
        self.inner.finish_thc()
    }

    /// Open `/coulomb` and write request/projection attributes.
    pub fn begin_coulomb(&mut self, begin: &MldumpCoulombBeginV1) -> Result<(), IoError> {
        self.inner.begin_coulomb(begin)
    }

    /// Write one Coulomb $q$ record immediately.
    pub fn write_coulomb_q(
        &mut self,
        record: &MldumpCoulombQRecordRefV1<'_>,
    ) -> Result<(), IoError> {
        self.inner.write_coulomb_q(record)
    }

    /// Close `/coulomb` after every mesh $q$ record has been written.
    pub fn finish_coulomb(&mut self) -> Result<(), IoError> {
        self.inner.finish_coulomb()
    }

    /// Finish the populated scalar file after all four sections.
    pub fn finish(self) -> Result<(), IoError> {
        self.inner.finish()
    }
}

impl SpinorMldumpStreamV1 {
    pub(crate) fn new(file: File, header: MldumpHeaderV1) -> Self {
        Self {
            inner: MldumpStreamV1::new(file, header),
        }
    }

    /// Open `/orbitals` and write shared attributes. $k$ records follow.
    pub fn begin_orbitals(&mut self, begin: &SpinorOrbitalsBeginV1) -> Result<(), IoError> {
        self.inner.begin_orbitals(begin)
    }

    /// Write one $k$ orbital record immediately.
    pub fn write_orbital_k(&mut self, record: &SpinorOrbitalKRefV1<'_>) -> Result<(), IoError> {
        self.inner.write_orbital_k(record)
    }

    /// Close `/orbitals` after every $k$ record has been written.
    pub fn finish_orbitals(&mut self) -> Result<(), IoError> {
        self.inner.finish_orbitals()
    }

    /// Open `/products` and write shared partition binding.
    pub fn begin_products(&mut self, begin: &SpinorProductsBeginV1<'_>) -> Result<(), IoError> {
        self.inner.begin_products(begin)
    }

    /// Write one site radial record immediately.
    pub fn write_product_site(
        &mut self,
        record: &SpinorProductSiteRefV1<'_>,
    ) -> Result<(), IoError> {
        self.inner.write_product_site(record)
    }

    /// Write one positional product $q$ record immediately.
    pub fn write_product_q(
        &mut self,
        record: &SpinorProductQRecordRefV1<'_>,
    ) -> Result<(), IoError> {
        self.inner.write_product_q(record)
    }

    /// Close `/products` after every site and $q$ record has been written.
    pub fn finish_products(&mut self) -> Result<(), IoError> {
        self.inner.finish_products()
    }

    /// Open `/thc` and write the shared parent grid and selection.
    pub fn begin_thc(&mut self, begin: &MldumpThcBeginV1<'_>) -> Result<(), IoError> {
        self.inner.begin_thc(begin)
    }

    /// Write one THC $q$ record immediately.
    pub fn write_thc_q(&mut self, record: &MldumpThcQRecordRefV1<'_>) -> Result<(), IoError> {
        self.inner.write_thc_q(record)
    }

    /// Close `/thc` after every mesh $q$ record has been written.
    pub fn finish_thc(&mut self) -> Result<(), IoError> {
        self.inner.finish_thc()
    }

    /// Open `/coulomb` and write request/projection attributes.
    pub fn begin_coulomb(&mut self, begin: &MldumpCoulombBeginV1) -> Result<(), IoError> {
        self.inner.begin_coulomb(begin)
    }

    /// Write one Coulomb $q$ record immediately.
    pub fn write_coulomb_q(
        &mut self,
        record: &MldumpCoulombQRecordRefV1<'_>,
    ) -> Result<(), IoError> {
        self.inner.write_coulomb_q(record)
    }

    /// Close `/coulomb` after every mesh $q$ record has been written.
    pub fn finish_coulomb(&mut self) -> Result<(), IoError> {
        self.inner.finish_coulomb()
    }

    /// Finish the populated spinor file after all four sections.
    pub fn finish(self) -> Result<(), IoError> {
        self.inner.finish()
    }
}
