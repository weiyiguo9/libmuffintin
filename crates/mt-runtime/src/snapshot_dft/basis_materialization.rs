use super::*;

impl SnapshotDftPhysics {
    pub(super) fn scalar_linearization_energies(
        &self,
        basis: &ScfBasis,
        site: &str,
        spin: usize,
    ) -> Result<Vec<Hartree>, SnapshotDftError> {
        (0..=basis.l_max)
            .map(|l| {
                let channels = basis
                    .resolved_channels
                    .iter()
                    .filter(|resolved| {
                        resolved.recipe.site == site
                            && resolved.recipe.treatment == ScfChannelTreatment::Valence
                            && channel_l(resolved.recipe.identity) == l
                    })
                    .collect::<Vec<_>>();
                let scalar = channels
                    .iter()
                    .filter(|resolved| {
                        matches!(resolved.recipe.identity, ScfChannelIdentity::ScalarL { .. })
                    })
                    .copied()
                    .collect::<Vec<_>>();
                if scalar.len() == 1 && channels.len() == 1 {
                    return Ok(spin_resolved_energy(scalar[0], spin));
                }
                if scalar.is_empty() && !channels.is_empty() {
                    let n = channel_n(channels[0].recipe.identity);
                    if channels
                        .iter()
                        .any(|resolved| channel_n(resolved.recipe.identity) != n)
                    {
                        return Err(SnapshotDftError::AmbiguousBaseChannel {
                            site: site.to_owned(),
                            l,
                        });
                    }
                    let partners = channels
                        .iter()
                        .map(|resolved| match resolved.recipe.identity {
                            ScfChannelIdentity::Kappa { kappa, .. } => {
                                Ok((Kappa::new(kappa)?, resolved.energy))
                            }
                            ScfChannelIdentity::ScalarL { .. } => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, SnapshotDftError>>()?;
                    return kappa_degeneracy_average(l, &partners).map_err(|source| {
                        SnapshotDftError::ScalarKappaAverage {
                            site: site.to_owned(),
                            l,
                            source,
                        }
                    });
                }
                if channels.is_empty() {
                    Err(SnapshotDftError::MissingMaterializedBaseChannel {
                        site: site.to_owned(),
                        l,
                    })
                } else {
                    Err(SnapshotDftError::AmbiguousBaseChannel {
                        site: site.to_owned(),
                        l,
                    })
                }
            })
            .collect()
    }

    pub(super) fn scalar_local_orbitals(
        &self,
        basis: &ScfBasis,
        site: &str,
        spin: usize,
    ) -> Result<Vec<ScalarLocalOrbitalRequest>, SnapshotDftError> {
        basis
            .resolved_channels
            .iter()
            .filter(|resolved| {
                resolved.recipe.site == site
                    && matches!(resolved.recipe.identity, ScfChannelIdentity::ScalarL { .. })
                    && matches!(
                        resolved.recipe.treatment,
                        ScfChannelTreatment::Lo | ScfChannelTreatment::Hdlo
                    )
            })
            .map(|resolved| {
                let l = channel_l(resolved.recipe.identity);
                Ok(match resolved.recipe.treatment {
                    ScfChannelTreatment::Lo => ScalarLocalOrbitalRequest::Lo {
                        l,
                        energy: spin_resolved_energy(resolved, spin),
                    },
                    ScfChannelTreatment::Hdlo => ScalarLocalOrbitalRequest::Hdlo { l },
                    ScfChannelTreatment::Core | ScfChannelTreatment::Valence => unreachable!(),
                })
            })
            .collect()
    }

    pub(super) fn spinor_linearization_energies(
        &self,
        basis: &ScfBasis,
        site: &str,
    ) -> Result<Vec<SpinorLinearizationEnergy>, SnapshotDftError> {
        let mut energies = Vec::new();
        for l in 0..=basis.l_max {
            let channels = basis
                .resolved_channels
                .iter()
                .filter(|resolved| {
                    resolved.recipe.site == site
                        && resolved.recipe.treatment == ScfChannelTreatment::Valence
                        && channel_l(resolved.recipe.identity) == l
                })
                .collect::<Vec<_>>();
            let scalar = channels
                .iter()
                .filter(|resolved| {
                    matches!(resolved.recipe.identity, ScfChannelIdentity::ScalarL { .. })
                })
                .copied()
                .collect::<Vec<_>>();
            if scalar.len() == 1 && channels.len() == 1 {
                for kappa in spinor_kappas_for_l(l)? {
                    energies.push(SpinorLinearizationEnergy {
                        kappa,
                        energy: scalar_component_energy(scalar[0], kappa),
                    });
                }
                continue;
            }
            if scalar.is_empty() && !channels.is_empty() {
                for kappa in spinor_kappas_for_l(l)? {
                    let matches = channels
                        .iter()
                        .filter(|resolved| {
                            matches!(
                                resolved.recipe.identity,
                                ScfChannelIdentity::Kappa {
                                    kappa: candidate,
                                    ..
                                } if candidate == kappa.get()
                            )
                        })
                        .copied()
                        .collect::<Vec<_>>();
                    if matches.len() != 1 {
                        return Err(SnapshotDftError::MissingSpinorBaseChannel {
                            site: site.to_owned(),
                            l,
                            kappa: kappa.get(),
                        });
                    }
                    energies.push(SpinorLinearizationEnergy {
                        kappa,
                        energy: matches[0].energy,
                    });
                }
                continue;
            }
            return Err(if channels.is_empty() {
                SnapshotDftError::MissingMaterializedBaseChannel {
                    site: site.to_owned(),
                    l,
                }
            } else {
                SnapshotDftError::AmbiguousBaseChannel {
                    site: site.to_owned(),
                    l,
                }
            });
        }
        Ok(energies)
    }

    pub(super) fn spinor_local_orbitals(
        &self,
        basis: &ScfBasis,
        site: &str,
    ) -> Result<Vec<SpinorLocalOrbitalRequest>, SnapshotDftError> {
        let mut orbitals = Vec::new();
        for resolved in basis.resolved_channels.iter().filter(|resolved| {
            resolved.recipe.site == site
                && matches!(
                    resolved.recipe.treatment,
                    ScfChannelTreatment::Lo | ScfChannelTreatment::Hdlo
                )
        }) {
            let kappas = match resolved.recipe.identity {
                ScfChannelIdentity::ScalarL { l, .. } => spinor_kappas_for_l(l)?,
                ScfChannelIdentity::Kappa { kappa, .. } => vec![Kappa::new(kappa)?],
            };
            for kappa in kappas {
                orbitals.push(match resolved.recipe.treatment {
                    ScfChannelTreatment::Lo => SpinorLocalOrbitalRequest::Lo {
                        kappa,
                        energy: scalar_component_energy(resolved, kappa),
                    },
                    ScfChannelTreatment::Hdlo => SpinorLocalOrbitalRequest::Hdlo { kappa },
                    ScfChannelTreatment::Core | ScfChannelTreatment::Valence => unreachable!(),
                });
            }
        }
        Ok(orbitals)
    }

    pub(super) fn materialize_current_basis(
        &self,
        iteration: usize,
        potential: &RegionalPotential,
        basis: &ScfBasis,
    ) -> Result<ScfBasis, SnapshotDftError> {
        let context = self
            .core_potentials
            .get(&iteration)
            .ok_or(SnapshotDftError::MissingCoreContinuation(iteration))?;
        let meshes = self.channel_meshes(basis)?;
        let extended = build_extended_core_potentials(
            &context.electrostatic,
            &context.exchange_correlation,
            &context.density,
            &meshes,
            context.spec,
        )?;
        self.materialize_nonspectral_basis(potential, basis, &extended)
    }

    pub(crate) fn materialize_nonspectral_basis(
        &self,
        potential: &RegionalPotential,
        requested: &ScfBasis,
        extended: &[muffintin_dft::BuiltExtendedCorePotential],
    ) -> Result<ScfBasis, SnapshotDftError> {
        self.require_potential_site_count(potential)?;
        let mut basis = requested.clone();
        basis.resolved_channels.clear();
        let mut lo_ordinals = BTreeMap::<(String, u32), usize>::new();
        for recipe in requested
            .channels
            .iter()
            .filter(|recipe| recipe.treatment != ScfChannelTreatment::Core)
        {
            let site_index = self.site_index(&recipe.site)?;
            let lo_ordinal = if recipe.treatment == ScfChannelTreatment::Lo {
                let key = (recipe.site.clone(), channel_l(recipe.identity));
                let ordinal = lo_ordinals.entry(key).or_default();
                let current = *ordinal;
                *ordinal += 1;
                Some(current)
            } else {
                None
            };
            let generated = if matches!(
                recipe.generator,
                LinearizationEnergyGenerator::BandCog | LinearizationEnergyGenerator::FermiOffset
            ) {
                self.provisional_spectral_channel(recipe, lo_ordinal)?
            } else {
                self.materialize_potential_channel(
                    recipe,
                    site_index,
                    potential,
                    &extended[site_index].potential,
                    lo_ordinal,
                )?
            };
            basis.resolved_channels.push(generated);
        }
        Ok(basis)
    }

    fn materialize_potential_channel(
        &self,
        recipe: &ScfChannelRecipe,
        site_index: usize,
        potential: &RegionalPotential,
        extended: &ExtendedCorePotential,
        lo_ordinal: Option<usize>,
    ) -> Result<ScfResolvedChannelEnergy, SnapshotDftError> {
        let site = &self.sites[site_index];
        let l = channel_l(recipe.identity);
        let one = |generated: GeneratedLinearizationEnergy| ScfResolvedChannelEnergy {
            recipe: recipe.clone(),
            energy: generated.energy,
            components: vec![generated],
        };
        let generated = match recipe.generator {
            LinearizationEnergyGenerator::Explicit => {
                let seed = recipe
                    .seed
                    .ok_or_else(|| SnapshotDftError::MissingChannelSeed {
                        site: recipe.site.clone(),
                        identity: recipe.identity,
                        generator: recipe.generator,
                    })?;
                return generate_explicit_energy(seed)
                    .map(one)
                    .map_err(|source| channel_generator_error(recipe, source));
            }
            LinearizationEnergyGenerator::FrozenSnapshot => {
                let up = self.snapshot_anchor_spin(recipe, lo_ordinal, 0)?;
                let down = self.snapshot_anchor_spin(recipe, lo_ordinal, 1)?;
                let mut components = vec![
                    generate_frozen_snapshot_energy(up)
                        .map_err(|source| channel_generator_error(recipe, source))?,
                ];
                let energy = if site.nonmagnetic_scalar {
                    up
                } else {
                    components.push(
                        generate_frozen_snapshot_energy(down)
                            .map_err(|source| channel_generator_error(recipe, source))?,
                    );
                    Hartree(0.5 * (up.get() + down.get()))
                };
                return Ok(ScfResolvedChannelEnergy {
                    recipe: recipe.clone(),
                    energy,
                    components,
                });
            }
            LinearizationEnergyGenerator::Atomic => {
                let kappas = channel_kappas(recipe.identity)?;
                let mut components = Vec::with_capacity(kappas.len());
                let mut partner_energies = Vec::with_capacity(kappas.len());
                for kappa in kappas {
                    let state = CoreState::new(channel_n(recipe.identity), kappa)?;
                    let generated = generate_atomic_energy(
                        &extended.mesh,
                        &extended.values,
                        AtomicEnergyRequest::new(
                            state,
                            self.nuclear_charges[site_index],
                            site.radius,
                        ),
                    )
                    .map_err(|source| channel_generator_error(recipe, source))?;
                    partner_energies.push((kappa, generated.energy));
                    components.push(generated);
                }
                let energy = match recipe.identity {
                    ScfChannelIdentity::ScalarL { .. } => {
                        kappa_degeneracy_average(l, &partner_energies)
                            .map_err(|source| channel_generator_error(recipe, source))?
                    }
                    ScfChannelIdentity::Kappa { .. } => components[0].energy,
                };
                return Ok(ScfResolvedChannelEnergy {
                    recipe: recipe.clone(),
                    energy,
                    components,
                });
            }
            LinearizationEnergyGenerator::BandCenter
            | LinearizationEnergyGenerator::LogDerivative => {
                if matches!(recipe.identity, ScfChannelIdentity::Kappa { .. }) {
                    return Err(SnapshotDftError::ScalarGeneratorRequiresLIdentity {
                        site: recipe.site.clone(),
                        identity: recipe.identity,
                        generator: recipe.generator,
                    });
                }
                let seed = match recipe.seed {
                    Some(seed) => seed,
                    None => self.snapshot_anchor(recipe, lo_ordinal)?,
                };
                let spherical = spherical_scalar_potential(potential, site_index, &site.id)?;
                if recipe.generator == LinearizationEnergyGenerator::BandCenter {
                    generate_band_center_energy(
                        &site.up.mesh,
                        &spherical,
                        RadialEquation::ScalarKoellingHarmon,
                        l,
                        seed,
                    )
                } else {
                    generate_log_derivative_energy(
                        &site.up.mesh,
                        &spherical,
                        RadialEquation::ScalarKoellingHarmon,
                        channel_n(recipe.identity),
                        l,
                        seed,
                        InverseBohr(-(f64::from(l) + 1.0) / site.radius.get()),
                    )
                }
            }
            LinearizationEnergyGenerator::BandCog | LinearizationEnergyGenerator::FermiOffset => {
                unreachable!("spectral generators are materialized after occupations")
            }
        };
        generated
            .map(one)
            .map_err(|source| channel_generator_error(recipe, source))
    }

    fn provisional_spectral_channel(
        &self,
        recipe: &ScfChannelRecipe,
        lo_ordinal: Option<usize>,
    ) -> Result<ScfResolvedChannelEnergy, SnapshotDftError> {
        let energy = self.snapshot_anchor(recipe, lo_ordinal)?;
        Ok(ScfResolvedChannelEnergy {
            recipe: recipe.clone(),
            energy,
            components: vec![GeneratedLinearizationEnergy {
                generator: recipe.generator,
                seed: Some(energy),
                energy,
                diagnostic: LinearizationEnergyDiagnostic::Stored,
            }],
        })
    }

    fn snapshot_anchor(
        &self,
        recipe: &ScfChannelRecipe,
        lo_ordinal: Option<usize>,
    ) -> Result<Hartree, SnapshotDftError> {
        let site_index = self.site_index(&recipe.site)?;
        let up = self.snapshot_anchor_spin(recipe, lo_ordinal, 0)?;
        let down = self.snapshot_anchor_spin(recipe, lo_ordinal, 1)?;
        Ok(if self.sites[site_index].nonmagnetic_scalar {
            up
        } else {
            Hartree(0.5 * (up.get() + down.get()))
        })
    }

    fn snapshot_anchor_spin(
        &self,
        recipe: &ScfChannelRecipe,
        lo_ordinal: Option<usize>,
        spin: usize,
    ) -> Result<Hartree, SnapshotDftError> {
        let site_index = self.site_index(&recipe.site)?;
        let site = &self.sites[site_index];
        let l = channel_l(recipe.identity);
        let radial = if spin == 0 { &site.up } else { &site.down };
        match recipe.treatment {
            ScfChannelTreatment::Lo => {
                let ordinal =
                    lo_ordinal.ok_or_else(|| SnapshotDftError::MissingFrozenSnapshotAnchor {
                        site: recipe.site.clone(),
                        identity: recipe.identity,
                        treatment: recipe.treatment,
                    })?;
                radial
                    .local_orbitals
                    .iter()
                    .filter(|(candidate_l, _)| *candidate_l == l)
                    .nth(ordinal)
                    .map(|(_, energy)| *energy)
                    .ok_or_else(|| SnapshotDftError::MissingFrozenSnapshotLo {
                        site: site.id.clone(),
                        l,
                        ordinal,
                        spin,
                    })
            }
            ScfChannelTreatment::Core
            | ScfChannelTreatment::Valence
            | ScfChannelTreatment::Hdlo => radial.linearization.get(&l).copied().ok_or_else(|| {
                SnapshotDftError::MissingFrozenSnapshotBase {
                    site: site.id.clone(),
                    l,
                    spin,
                }
            }),
        }
    }

    pub(crate) fn channel_meshes(
        &self,
        basis: &ScfBasis,
    ) -> Result<Vec<ExponentialMesh>, SnapshotDftError> {
        self.sites
            .iter()
            .enumerate()
            .map(|(site_index, site)| {
                let maximum_n = basis
                    .channels
                    .iter()
                    .filter(|recipe| recipe.site == site.id)
                    .map(|recipe| channel_n(recipe.identity))
                    .max()
                    .unwrap_or(1);
                let orbital_scale =
                    f64::from(maximum_n).powi(2) / self.nuclear_charges[site_index].max(1.0);
                let outer_radius = (4.0 * site.radius.get()).max(40.0 * orbital_scale);
                extend_mesh(&site.up.mesh, outer_radius)
            })
            .collect()
    }
}
