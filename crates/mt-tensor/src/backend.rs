//! Backend-neutral Einstein summation.
//!
//! Physics code calls [`einsum`]. The default implementation is RSTSR 0.7.10
//! with TBLIS (`rt::tblis::einsum`). The same subscripts are the contract for
//! an optional tenferro-rs backend.

use crate::{Axis, ComplexTensor, TensorError};

/// Executes `einsum` for one local tensor backend.
pub trait EinsumBackend {
    fn name() -> &'static str;
    fn einsum(subscripts: &str, operands: &[&ComplexTensor]) -> Result<ComplexTensor, TensorError>;
}

/// Einstein summation over dense complex tensors.
///
/// Subscripts follow NumPy form, for example `ci,cd,dj->ij`. Conjugation is
/// applied to an operand before the call; it is not encoded in the subscript.
/// Output axes are inferred from the operand axis roles bound to each label.
pub fn einsum(subscripts: &str, operands: &[&ComplexTensor]) -> Result<ComplexTensor, TensorError> {
    #[cfg(feature = "backend-rstsr")]
    {
        crate::rstsr_tblis::RstsrTblisBackend::einsum(subscripts, operands)
    }
    #[cfg(not(feature = "backend-rstsr"))]
    {
        let _ = (subscripts, operands);
        Err(TensorError::Backend(
            "no einsum backend enabled; enable backend-rstsr or backend-tenferro".into(),
        ))
    }
}

pub fn active_backend_name() -> &'static str {
    #[cfg(feature = "backend-rstsr")]
    {
        crate::rstsr_tblis::RstsrTblisBackend::name()
    }
    #[cfg(not(feature = "backend-rstsr"))]
    {
        "none"
    }
}

/// Same subscripts as [`einsum`], evaluated by tenferro-rs.
///
/// Requires rustc 1.96 and the `backend-tenferro` feature. The workspace
/// MSRV remains 1.89; raise `rust-version` locally to compile this path.
#[cfg(feature = "backend-tenferro")]
pub fn einsum_tenferro(
    subscripts: &str,
    operands: &[&ComplexTensor],
) -> Result<ComplexTensor, TensorError> {
    crate::tenferro::einsum(subscripts, operands)
}

pub(crate) fn infer_output_axes(
    subscripts: &str,
    operands: &[&ComplexTensor],
) -> Result<Vec<Axis>, TensorError> {
    let (input_labels, output_labels) = parse_subscripts(subscripts)?;
    if input_labels.len() != operands.len() {
        return Err(TensorError::EinsumArity {
            subscripts: subscripts.to_string(),
            expected: input_labels.len(),
            actual: operands.len(),
        });
    }
    let mut bound = std::collections::BTreeMap::<char, Axis>::new();
    for (labels, tensor) in input_labels.iter().zip(operands) {
        if labels.len() != tensor.rank() {
            return Err(TensorError::Rank {
                expected: tensor.rank(),
                actual: labels.len(),
            });
        }
        for (index, label) in labels.iter().enumerate() {
            let axis = tensor.axes()[index];
            if let Some(previous) = bound.insert(*label, axis) {
                if previous != axis {
                    return Err(TensorError::Axis {
                        index,
                        expected: previous,
                        actual: axis,
                    });
                }
            }
        }
    }
    output_labels
        .iter()
        .map(|label| {
            bound
                .get(label)
                .copied()
                .ok_or_else(|| TensorError::EinsumLabel {
                    label: *label,
                    subscripts: subscripts.to_string(),
                })
        })
        .collect()
}

pub(crate) fn parse_subscripts(
    subscripts: &str,
) -> Result<(Vec<Vec<char>>, Vec<char>), TensorError> {
    let normalized: String = subscripts.chars().filter(|c| !c.is_whitespace()).collect();
    let Some((inputs, output)) = normalized.split_once("->") else {
        return Err(TensorError::EinsumSyntax {
            subscripts: subscripts.to_string(),
        });
    };
    if inputs.is_empty() {
        return Err(TensorError::EinsumSyntax {
            subscripts: subscripts.to_string(),
        });
    }
    let input_labels = inputs
        .split(',')
        .map(|part| part.chars().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    Ok((input_labels, output.chars().collect()))
}
