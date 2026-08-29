//! Spinor MLDUMP v1 payload DTO and streaming session.

use hdf5_metno::File;

use super::response::{
    CoulombAlignmentSummary, MldumpCoulombBeginV1, MldumpCoulombQRecordRefV1, MldumpCoulombV1,
    MldumpThcBeginV1, MldumpThcQRecordRefV1, MldumpThcV1, ProductAlignmentSummary,
    ThcAlignmentSummary,
};
use super::spinor_orbitals::{
    MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, SpinorOrbitalKRefV1, SpinorOrbitalsBeginV1,
    SpinorOrbitalsV1,
};
use super::spinor_products::{
    SpinorProductQRecordRefV1, SpinorProductSiteRefV1, SpinorProductsBeginV1, SpinorProductsV1,
};
use super::{MldumpHeaderV1, require_record_capacity, section_already_written, stream_state_error};
use crate::error::{IoError, ValidationError};

/// Owned spinor payload: orbitals, products, THC, and Coulomb. MPB is not a field.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorMldumpV1 {
    pub orbitals: SpinorOrbitalsV1,
    pub products: SpinorProductsV1,
    pub thc: MldumpThcV1,
    pub coulomb: MldumpCoulombV1,
}

/// Streaming spinor payload session. Large records are written immediately;
/// only small counters, $q$ bindings, pair-layout counts, and provenance
/// strings are retained.
#[derive(Debug)]
pub struct SpinorMldumpStreamV1 {
    file: File,
    header: MldumpHeaderV1,
    phase: SpinorStreamPhase,
    orbital_n_k: usize,
    orbitals_band_window: usize,
    orbitals_written: bool,
    product_summary: Option<ProductAlignmentSummary>,
    thc_summary: Option<ThcAlignmentSummary>,
    coulomb_summary: Option<CoulombAlignmentSummary>,
    products_n_site: usize,
    thc_n_parent: usize,
    thc_effective_rank: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpinorStreamPhase {
    Start,
    Orbitals { next_k: usize },
    Products { next_site: usize, next_q: usize },
    Thc { next_q: usize },
    Coulomb { next_q: usize },
}

impl SpinorMldumpStreamV1 {
    pub(crate) fn new(file: File, header: MldumpHeaderV1) -> Self {
        Self {
            orbital_n_k: header.mesh.k_points.len(),
            file,
            header,
            phase: SpinorStreamPhase::Start,
            orbitals_band_window: 0,
            orbitals_written: false,
            product_summary: None,
            thc_summary: None,
            coulomb_summary: None,
            products_n_site: 0,
            thc_n_parent: 0,
            thc_effective_rank: 0,
        }
    }

    /// Open `/orbitals` and write shared attributes. $k$ records follow.
    pub fn begin_orbitals(&mut self, begin: &SpinorOrbitalsBeginV1) -> Result<(), IoError> {
        self.require_idle("begin_orbitals")?;
        if self.orbitals_written {
            return Err(section_already_written("/orbitals"));
        }
        super::spinor_orbitals::begin_spinor_orbitals(&self.file, begin)?;
        self.orbitals_band_window = begin.band_window_count;
        self.orbitals_written = true;
        self.phase = SpinorStreamPhase::Orbitals { next_k: 0 };
        Ok(())
    }

    /// Write one $k$ orbital record immediately.
    pub fn write_orbital_k(&mut self, record: &SpinorOrbitalKRefV1<'_>) -> Result<(), IoError> {
        let next_k = match self.phase {
            SpinorStreamPhase::Orbitals { next_k } => next_k,
            _ => {
                return Err(stream_state_error(
                    "spinor",
                    "write_orbital_k",
                    "orbitals section in progress",
                ));
            }
        };
        require_record_capacity("orbitals.record", next_k, self.orbital_n_k)?;
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
            self.orbitals_band_window,
            record,
        )?;
        self.phase = SpinorStreamPhase::Orbitals { next_k: next_k + 1 };
        Ok(())
    }

    /// Close `/orbitals` after every $k$ record has been written.
    pub fn finish_orbitals(&mut self) -> Result<(), IoError> {
        let SpinorStreamPhase::Orbitals { next_k } = self.phase else {
            return Err(stream_state_error(
                "spinor",
                "finish_orbitals",
                "orbitals section in progress",
            ));
        };
        if next_k != self.orbital_n_k {
            return Err(ValidationError::InvalidValue {
                path: "orbitals".to_owned(),
                expected: format!("{} k records", self.orbital_n_k),
                actual: format!("next k={next_k}"),
            }
            .into());
        }
        self.phase = SpinorStreamPhase::Start;
        Ok(())
    }

    /// Open `/products` and write shared partition binding.
    pub fn begin_products(&mut self, begin: &SpinorProductsBeginV1<'_>) -> Result<(), IoError> {
        self.require_idle("begin_products")?;
        if self.product_summary.is_some() {
            return Err(section_already_written("/products"));
        }
        super::spinor_products::begin_spinor_products(&self.file, &self.header, begin)?;
        self.products_n_site = self.header.geometry.sites.len();
        self.product_summary = Some(ProductAlignmentSummary::new(begin.n_k, begin.n_orb));
        self.phase = SpinorStreamPhase::Products {
            next_site: 0,
            next_q: 0,
        };
        Ok(())
    }

    /// Write one site radial record immediately.
    pub fn write_product_site(
        &mut self,
        record: &SpinorProductSiteRefV1<'_>,
    ) -> Result<(), IoError> {
        let next_site = match self.phase {
            SpinorStreamPhase::Products {
                next_site,
                next_q: 0,
            } => next_site,
            _ => {
                return Err(stream_state_error(
                    "spinor",
                    "write_product_site",
                    "product sites before q records",
                ));
            }
        };
        require_record_capacity("products.sites", next_site, self.products_n_site)?;
        if record.site_index != next_site {
            return Err(ValidationError::InvalidValue {
                path: "products.sites".to_owned(),
                expected: next_site.to_string(),
                actual: record.site_index.to_string(),
            }
            .into());
        }
        super::spinor_products::write_spinor_product_site(
            &self.file,
            &self.header,
            next_site,
            record,
        )?;
        self.phase = SpinorStreamPhase::Products {
            next_site: next_site + 1,
            next_q: 0,
        };
        Ok(())
    }

    /// Write one positional product $q$ record immediately.
    pub fn write_product_q(
        &mut self,
        record: &SpinorProductQRecordRefV1<'_>,
    ) -> Result<(), IoError> {
        let (next_site, next_q) = match self.phase {
            SpinorStreamPhase::Products { next_site, next_q } => (next_site, next_q),
            _ => {
                return Err(stream_state_error(
                    "spinor",
                    "write_product_q",
                    "products section in progress",
                ));
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
        if record.q_index != next_q {
            return Err(ValidationError::InvalidValue {
                path: "products.q_records".to_owned(),
                expected: next_q.to_string(),
                actual: record.q_index.to_string(),
            }
            .into());
        }
        super::spinor_products::write_spinor_product_q(&self.file, next_q, record)?;
        if let Some(summary) = self.product_summary.as_mut() {
            summary.push_q_binding(
                record.q_index,
                record.transfer_cartesian,
                record.global_transfer,
            );
        }
        self.phase = SpinorStreamPhase::Products {
            next_site,
            next_q: next_q + 1,
        };
        Ok(())
    }

    /// Close `/products` after every site and $q$ record has been written.
    pub fn finish_products(&mut self) -> Result<(), IoError> {
        let SpinorStreamPhase::Products { next_site, next_q } = self.phase else {
            return Err(stream_state_error(
                "spinor",
                "finish_products",
                "products section in progress",
            ));
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
        self.phase = SpinorStreamPhase::Start;
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
        super::response::begin_mldump_thc(
            &self.file,
            &self.header,
            begin,
            MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
        )?;
        self.thc_n_parent = begin.parent_grid.n_points;
        self.thc_effective_rank = begin.effective_rank;
        self.thc_summary = Some(ThcAlignmentSummary::new());
        self.phase = SpinorStreamPhase::Thc { next_q: 0 };
        Ok(())
    }

    /// Write one THC $q$ record immediately.
    pub fn write_thc_q(&mut self, record: &MldumpThcQRecordRefV1<'_>) -> Result<(), IoError> {
        let next_q = match self.phase {
            SpinorStreamPhase::Thc { next_q } => next_q,
            _ => {
                return Err(stream_state_error(
                    "spinor",
                    "write_thc_q",
                    "thc section in progress",
                ));
            }
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
        self.phase = SpinorStreamPhase::Thc { next_q: next_q + 1 };
        Ok(())
    }

    /// Close `/thc` after every mesh $q$ record has been written.
    pub fn finish_thc(&mut self) -> Result<(), IoError> {
        let SpinorStreamPhase::Thc { next_q } = self.phase else {
            return Err(stream_state_error(
                "spinor",
                "finish_thc",
                "thc section in progress",
            ));
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
        self.phase = SpinorStreamPhase::Start;
        Ok(())
    }

    /// Open `/coulomb` and write request/projection attributes.
    pub fn begin_coulomb(&mut self, begin: &MldumpCoulombBeginV1) -> Result<(), IoError> {
        self.require_idle("begin_coulomb")?;
        if self.coulomb_summary.is_some() {
            return Err(section_already_written("/coulomb"));
        }
        super::response::begin_mldump_coulomb(
            &self.file,
            begin,
            MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
        )?;
        self.coulomb_summary = Some(CoulombAlignmentSummary::new());
        self.phase = SpinorStreamPhase::Coulomb { next_q: 0 };
        Ok(())
    }

    /// Write one Coulomb $q$ record immediately.
    pub fn write_coulomb_q(
        &mut self,
        record: &MldumpCoulombQRecordRefV1<'_>,
    ) -> Result<(), IoError> {
        let next_q = match self.phase {
            SpinorStreamPhase::Coulomb { next_q } => next_q,
            _ => {
                return Err(stream_state_error(
                    "spinor",
                    "write_coulomb_q",
                    "coulomb section in progress",
                ));
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
        self.phase = SpinorStreamPhase::Coulomb { next_q: next_q + 1 };
        Ok(())
    }

    /// Close `/coulomb` after every mesh $q$ record has been written.
    pub fn finish_coulomb(&mut self) -> Result<(), IoError> {
        let SpinorStreamPhase::Coulomb { next_q } = self.phase else {
            return Err(stream_state_error(
                "spinor",
                "finish_coulomb",
                "coulomb section in progress",
            ));
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
        self.phase = SpinorStreamPhase::Start;
        Ok(())
    }

    /// Finish the populated spinor file after all four sections.
    pub fn finish(self) -> Result<(), IoError> {
        if self.phase != SpinorStreamPhase::Start {
            return Err(stream_state_error(
                "spinor",
                "finish",
                "no section left open",
            ));
        }
        if !self.orbitals_written {
            return Err(ValidationError::InvalidValue {
                path: "spinor".to_owned(),
                expected: "alignment summaries for all four written sections".to_owned(),
                actual: "missing orbitals summary".to_owned(),
            }
            .into());
        }
        match (
            self.product_summary.as_ref(),
            self.thc_summary.as_ref(),
            self.coulomb_summary.as_ref(),
        ) {
            (Some(products), Some(thc), Some(coulomb)) => {
                super::response::validate_payload_alignment(
                    &self.header,
                    self.orbitals_band_window,
                    products,
                    thc,
                    coulomb,
                )
            }
            _ => Err(ValidationError::InvalidValue {
                path: "spinor".to_owned(),
                expected: "alignment summaries for all four written sections".to_owned(),
                actual: "missing retained summary".to_owned(),
            }
            .into()),
        }
    }

    fn require_idle(&self, method: &str) -> Result<(), IoError> {
        if self.phase == SpinorStreamPhase::Start {
            Ok(())
        } else {
            Err(stream_state_error(
                "spinor",
                method,
                "no section currently open",
            ))
        }
    }
}
