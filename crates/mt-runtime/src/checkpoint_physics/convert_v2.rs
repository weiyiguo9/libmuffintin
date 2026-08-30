use super::*;

/// Build a validated V2 restart checkpoint from a converged SCF state.
///
/// `template` supplies the immutable cell, sites, radial equations, and
/// linearization metadata. The state supplies the complete noncollinear
/// density and potential without reducing their Cartesian Pauli components.
pub fn checkpoint_v2_from_state(
    template: &CheckpointV2,
    state: &ScfState,
) -> Result<CheckpointV2, CheckpointPhysicsError> {
    template.validate()?;
    let template_potential = match &template.initial {
        InitialV2::FrozenPotential { potential } | InitialV2::Restart { potential, .. } => {
            potential
        }
    };
    let mut potential_hints = template_potential.basis_hints;
    potential_hints.plane_wave_cutoff = Some(state.basis.plane_wave_cutoff);
    let mut density_hints = match &template.initial {
        InitialV2::Restart { density, .. } => density.basis_hints,
        InitialV2::FrozenPotential { .. } => template_potential.basis_hints,
    };
    density_hints.plane_wave_cutoff = Some(state.basis.plane_wave_cutoff);
    let angular_basis = template.meta.potential_convention.angular_basis;
    let density = DensityV2 {
        unit: FieldUnitV2::BohrMinus3,
        representation: FieldRepresentationV2::PeriodicExtension,
        angular_basis,
        basis_hints: density_hints,
        n: regional_scalar_to_v2(
            state.density.charge(),
            &template.geometry.sites,
            angular_basis,
        )?,
        mx: regional_scalar_to_v2(
            &state.density.magnetization()[0],
            &template.geometry.sites,
            angular_basis,
        )?,
        my: regional_scalar_to_v2(
            &state.density.magnetization()[1],
            &template.geometry.sites,
            angular_basis,
        )?,
        mz: regional_scalar_to_v2(
            &state.density.magnetization()[2],
            &template.geometry.sites,
            angular_basis,
        )?,
    };
    let potential = PotentialV2 {
        unit: FieldUnitV2::Hartree,
        representation: FieldRepresentationV2::MaskedOperator,
        angular_basis,
        basis_hints: potential_hints,
        v0: regional_scalar_to_v2(
            state.potential.scalar(),
            &template.geometry.sites,
            angular_basis,
        )?,
        bx: regional_scalar_to_v2(
            &state.potential.magnetic()[0],
            &template.geometry.sites,
            angular_basis,
        )?,
        by: regional_scalar_to_v2(
            &state.potential.magnetic()[1],
            &template.geometry.sites,
            angular_basis,
        )?,
        bz: regional_scalar_to_v2(
            &state.potential.magnetic()[2],
            &template.geometry.sites,
            angular_basis,
        )?,
    };
    let checkpoint = CheckpointV2::new(
        template.meta.clone(),
        template.geometry.clone(),
        InitialV2::Restart { density, potential },
    );
    checkpoint.validate()?;
    Ok(checkpoint)
}

pub(super) fn convert_v2_site_bases(
    site_id: &str,
    bases: &[muffintin_io::SiteRadialBasisV2],
) -> Result<(CheckpointSpin, CheckpointSpin, bool), CheckpointPhysicsError> {
    let scalar = bases
        .iter()
        .find(|basis| basis.site_id == site_id && basis.spin == RadialBasisSpinV2::Scalar);
    let up = bases
        .iter()
        .find(|basis| basis.site_id == site_id && basis.spin == RadialBasisSpinV2::Up);
    let down = bases
        .iter()
        .find(|basis| basis.site_id == site_id && basis.spin == RadialBasisSpinV2::Down);
    match (scalar, up, down) {
        (Some(scalar), None, None) => {
            let converted = convert_v2_radial_basis(scalar)?;
            Ok((converted.clone(), converted, true))
        }
        (None, Some(up), Some(down)) => Ok((
            convert_v2_radial_basis(up)?,
            convert_v2_radial_basis(down)?,
            false,
        )),
        _ => Err(CheckpointPhysicsError::InvalidRadialBasisSpins {
            site: site_id.to_owned(),
        }),
    }
}

fn convert_v2_radial_basis(
    basis: &muffintin_io::SiteRadialBasisV2,
) -> Result<CheckpointSpin, CheckpointPhysicsError> {
    let mesh = ExponentialMesh::new(
        Bohr(basis.mesh.first),
        basis.mesh.log_increment,
        basis.mesh.point_count,
    )?;
    Ok(CheckpointSpin {
        route: match basis.radial_equation {
            RadialEquationTag::Schroedinger => RadialRoute::Schroedinger,
            RadialEquationTag::ScalarKoellingHarmon => RadialRoute::ScalarKoellingHarmon,
            RadialEquationTag::FullyRelativisticDirac => RadialRoute::Dirac,
        },
        mesh,
        linearization: basis
            .linearization
            .linearization_energies
            .iter()
            .map(|parameter| (parameter.l, Hartree(parameter.energy)))
            .collect(),
        local_orbitals: basis
            .linearization
            .local_orbital_energies
            .iter()
            .map(|parameter| (parameter.l, Hartree(parameter.energy)))
            .collect(),
    })
}

pub(super) fn regional_potential_from_v2(
    potential: &PotentialV2,
    geometry: &InterstitialGeometry,
    sites: &[CheckpointSite],
    reciprocal: ReciprocalLattice,
) -> Result<RegionalPotential, CheckpointPhysicsError> {
    let scalar = regional_scalar_from_v2(
        &potential.v0,
        potential.angular_basis,
        geometry,
        sites,
        reciprocal,
    )?;
    let magnetic = [&potential.bx, &potential.by, &potential.bz]
        .map(|field| {
            regional_scalar_from_v2(field, potential.angular_basis, geometry, sites, reciprocal)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .expect("three V2 magnetic components remain three components");
    Ok(RegionalPotential::new(scalar, magnetic)?)
}

pub(super) fn regional_density_from_v2(
    density: &DensityV2,
    geometry: &InterstitialGeometry,
    sites: &[CheckpointSite],
    reciprocal: ReciprocalLattice,
) -> Result<RegionalDensity, CheckpointPhysicsError> {
    let charge = regional_scalar_from_v2(
        &density.n,
        density.angular_basis,
        geometry,
        sites,
        reciprocal,
    )?;
    let magnetization = [&density.mx, &density.my, &density.mz]
        .map(|field| {
            regional_scalar_from_v2(field, density.angular_basis, geometry, sites, reciprocal)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .expect("three V2 magnetization components remain three components");
    Ok(RegionalDensity::new(charge, magnetization)?)
}

fn regional_scalar_from_v2(
    field: &RegionalFieldV2,
    angular_basis: AngularBasis,
    geometry: &InterstitialGeometry,
    sites: &[CheckpointSite],
    reciprocal: ReciprocalLattice,
) -> Result<RegionalScalarField, CheckpointPhysicsError> {
    let convention = match angular_basis {
        AngularBasis::ComplexCondonShortley => HarmonicConvention::Complex,
        AngularBasis::RealTesseralCondonShortley => HarmonicConvention::Real,
    };
    let by_site = field
        .muffin_tins
        .iter()
        .map(|field| (field.site_id.as_str(), field))
        .collect::<BTreeMap<_, _>>();
    let muffin_tins = sites
        .iter()
        .map(|site| {
            let source = by_site
                .get(site.id.as_str())
                .ok_or_else(|| CheckpointPhysicsError::MissingV2FieldSite(site.id.clone()))?;
            let channels = source.channels.iter().map(|channel| {
                let scale = if (channel.l, channel.m) == (0, 0) {
                    (4.0 * PI).sqrt()
                } else {
                    1.0
                };
                let values = channel
                    .real
                    .iter()
                    .enumerate()
                    .map(|(index, &real)| {
                        Complex64::new(
                            scale * real,
                            scale * channel.imaginary.get(index).copied().unwrap_or(0.0),
                        )
                    })
                    .collect();
                ((channel.l, channel.m), values)
            });
            Ok(MuffinTinField::new(
                site.up.mesh.clone(),
                SphereField::new(convention, channels)?,
            )?)
        })
        .collect::<Result<Vec<_>, CheckpointPhysicsError>>()?;
    let mut coefficients = field.interstitial.coefficients.clone();
    coefficients.sort_by_key(|coefficient| coefficient.g);
    let vectors = coefficients
        .iter()
        .map(|coefficient| g_vector(reciprocal, coefficient.g))
        .collect();
    let layout = FourierLayout::new(reciprocal, vectors)?;
    if layout.index([0; 3]).is_none() {
        return Err(CheckpointPhysicsError::MissingInterstitialZero);
    }
    let values = coefficients
        .iter()
        .map(|coefficient| Complex64::new(coefficient.value.real, coefficient.value.imaginary))
        .collect();
    let interstitial =
        InterstitialField::from_fourier_field(HermitianFourierField::new(layout, values)?);
    Ok(RegionalScalarField::new(
        geometry.clone(),
        muffin_tins,
        interstitial,
    )?)
}

pub(super) fn regional_scalar_to_v2(
    field: &RegionalScalarField,
    sites: &[muffintin_io::SiteV2],
    angular_basis: AngularBasis,
) -> Result<RegionalFieldV2, CheckpointPhysicsError> {
    if field.muffin_tins().len() != sites.len() {
        return Err(CheckpointPhysicsError::ExportSiteCount {
            expected: sites.len(),
            actual: field.muffin_tins().len(),
        });
    }
    let muffin_tins = sites
        .iter()
        .zip(field.muffin_tins())
        .map(|(site, field)| {
            Ok(MuffinTinFieldV2 {
                site_id: site.id.clone(),
                channels: sphere_channels_to_v2(field.field(), angular_basis)?,
            })
        })
        .collect::<Result<Vec<_>, CheckpointPhysicsError>>()?;
    let coefficients = field
        .interstitial()
        .field()
        .iter()
        .map(|(vector, &value)| FourierCoefficientV2 {
            g: vector.index,
            value: Complex64V2 {
                real: value.re,
                imaginary: value.im,
            },
        })
        .collect();
    Ok(RegionalFieldV2 {
        muffin_tins,
        interstitial: InterstitialFieldV2 { coefficients },
    })
}

fn sphere_channels_to_v2(
    field: &SphereField,
    angular_basis: AngularBasis,
) -> Result<Vec<SphericalChannelV2>, CheckpointPhysicsError> {
    let target = match angular_basis {
        AngularBasis::ComplexCondonShortley => HarmonicConvention::Complex,
        AngularBasis::RealTesseralCondonShortley => HarmonicConvention::Real,
    };
    if field.convention() == target {
        return Ok(field
            .channels()
            .map(|(channel, values)| channel_to_v2(channel.l, channel.m, values))
            .collect());
    }
    if field.convention() == HarmonicConvention::Complex {
        return Err(CheckpointPhysicsError::UnsupportedAngularConversion {
            from: HarmonicConvention::Complex,
            target,
        });
    }

    let by_channel = field
        .channels()
        .map(|(channel, values)| ((channel.l, channel.m), values))
        .collect::<BTreeMap<_, _>>();
    let l_max = by_channel.keys().map(|(l, _)| *l).max().unwrap_or(0);
    let mut channels = Vec::with_capacity(by_channel.len());
    for l in 0..=l_max {
        if let Some(values) = by_channel.get(&(l, 0)) {
            channels.push(channel_to_v2(l, 0, values));
        }
        for q in 1..=l {
            let q = i32::try_from(q).expect("u32 angular momentum fits stored i32 m");
            let Some(positive) = by_channel.get(&(l, q)) else {
                if by_channel.contains_key(&(l, -q)) {
                    return Err(CheckpointPhysicsError::UnpairedRealTesseralChannel { l, m: -q });
                }
                continue;
            };
            let Some(negative) = by_channel.get(&(l, -q)) else {
                return Err(CheckpointPhysicsError::UnpairedRealTesseralChannel { l, m: q });
            };
            let phase = if q.unsigned_abs() % 2 == 0 { 1.0 } else { -1.0 };
            let scale = 1.0 / 2.0_f64.sqrt();
            let complex_positive = positive
                .iter()
                .zip(*negative)
                .map(|(&cosine, &sine)| scale * (phase * cosine + Complex64::i() * sine))
                .collect::<Vec<_>>();
            let complex_negative = positive
                .iter()
                .zip(*negative)
                .map(|(&cosine, &sine)| scale * (cosine - Complex64::i() * phase * sine))
                .collect::<Vec<_>>();
            channels.push(channel_to_v2(l, -q, &complex_negative));
            channels.push(channel_to_v2(l, q, &complex_positive));
        }
    }
    Ok(channels)
}

fn channel_to_v2(l: u32, m: i32, values: &[Complex64]) -> SphericalChannelV2 {
    let scale = if (l, m) == (0, 0) {
        1.0 / (4.0 * PI).sqrt()
    } else {
        1.0
    };
    SphericalChannelV2 {
        l,
        m,
        real: values.iter().map(|value| scale * value.re).collect(),
        imaginary: values.iter().map(|value| scale * value.im).collect(),
    }
}
