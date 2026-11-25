//! Shared validation, GGUF/SafeTensors alias resolution, matrix projection, and attention helpers.

use jero_core::{
    ArchRecipe, K3Error, ModelMetadata, ModelWeights, QuantType, SafeTensorsFile, TensorDescriptor,
    TensorView,
};

use crate::kernels_compat::{dequantize_token_row, matvec_any};

pub(crate) fn bounds_err(
    tensor_name: &str,
    expected: usize,
    actual: usize,
    reason: String,
) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: tensor_name.to_owned(),
        offset: 0,
        byte_len: expected.saturating_mul(size_of::<f32>()),
        buffer_len: actual.saturating_mul(size_of::<f32>()),
        reason,
    }
}

pub(crate) fn mismatch_err(arch: &str, reason: impl Into<String>) -> K3Error {
    K3Error::CapabilityMismatch {
        architecture: arch.to_owned(),
        reason: reason.into(),
    }
}

pub(crate) fn checked_mul(name: &str, a: usize, b: usize) -> Result<usize, K3Error> {
    a.checked_mul(b).ok_or_else(|| {
        bounds_err(
            name,
            usize::MAX,
            0,
            format!("{name} dimensions ({a} * {b}) overflowed usize"),
        )
    })
}

pub(crate) fn validate_finite_slice(name: &str, data: &[f32]) -> Result<(), K3Error> {
    for (idx, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: idx.saturating_mul(size_of::<f32>()),
                byte_len: size_of::<f32>(),
                buffer_len: size_of_val(data),
                reason: format!("non-finite f32 ({val}) in '{name}' at index {idx}"),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_raw_unquantized_finite(
    view: &TensorView<'_>,
    name: &str,
) -> Result<(), K3Error> {
    let total = view.num_elements();
    let raw = view.as_bytes();
    match view.quant() {
        QuantType::F32 => {
            let req = checked_mul(name, total, size_of::<f32>())?;
            if raw.len() < req {
                return Err(bounds_err(
                    name,
                    total,
                    raw.len() / size_of::<f32>(),
                    format!("F32 tensor '{name}' byte slice truncated"),
                ));
            }
            for (idx, chunk) in raw[..req].chunks_exact(size_of::<f32>()).enumerate() {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !val.is_finite() {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: name.to_owned(),
                        offset: idx.saturating_mul(size_of::<f32>()),
                        byte_len: size_of::<f32>(),
                        buffer_len: raw.len(),
                        reason: format!("non-finite f32 ({val}) in '{name}' at index {idx}"),
                    });
                }
            }
        }
        QuantType::F16 | QuantType::BF16 => {
            let req = checked_mul(name, total, 2)?;
            if raw.len() < req {
                return Err(bounds_err(
                    name,
                    total,
                    raw.len() / 2,
                    format!("{:?} tensor '{name}' byte slice truncated", view.quant()),
                ));
            }
            let exp_mask = if view.quant() == QuantType::F16 {
                0x7C00_u16
            } else {
                0x7F80_u16
            };
            for (idx, chunk) in raw[..req].chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                if (bits & exp_mask) == exp_mask {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: name.to_owned(),
                        offset: idx.saturating_mul(2),
                        byte_len: 2,
                        buffer_len: raw.len(),
                        reason: format!(
                            "non-finite {:?} value (0x{bits:04X}) in '{name}' at index {idx}",
                            view.quant()
                        ),
                    });
                }
            }
        }
        _ => {}
    }
    Ok(())
}

pub(crate) fn dequantize_all_finite(
    view: &TensorView<'_>,
    name: &str,
) -> Result<Vec<f32>, K3Error> {
    let total = view.num_elements();
    if total == 0 {
        return Err(bounds_err(
            name,
            1,
            0,
            format!("tensor '{name}' cannot be empty"),
        ));
    }
    validate_raw_unquantized_finite(view, name)?;
    let mut out = vec![0.0_f32; total];
    if view.quant() == QuantType::F32 {
        let raw = view.as_bytes();
        for (dst, chunk) in out.iter_mut().zip(raw.chunks_exact(size_of::<f32>())) {
            *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    } else {
        let row_len = match view.shape() {
            [cols, ..] if *cols > 0 => *cols,
            _ => total,
        };
        let row_count = total.checked_div(row_len).unwrap_or(1).max(1);
        for row_idx in 0..row_count {
            let start = row_idx * row_len;
            dequantize_token_row(view, row_idx, &mut out[start..start + row_len])?;
        }
    }
    validate_finite_slice(name, &out)?;
    Ok(out)
}

pub(crate) fn find_tensor_by_aliases<'m>(
    metadata: &'m ModelMetadata,
    aliases: &[&str],
) -> Option<&'m TensorDescriptor> {
    aliases.iter().find_map(|name| metadata.find_tensor(name))
}

pub(crate) fn contains_any_tensor(weights: &dyn ModelWeights, aliases: &[&str]) -> bool {
    aliases.iter().any(|name| weights.contains_tensor(name))
}

pub(crate) fn lookup_tensor_aliases<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
) -> Result<TensorView<'w>, K3Error> {
    for &name in aliases {
        if weights.contains_tensor(name) {
            return weights.tensor(name);
        }
    }
    weights.tensor(aliases.first().copied().unwrap_or("missing_tensor"))
}

pub(crate) fn lookup_tensor_owned_aliases<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[String],
) -> Result<TensorView<'w>, K3Error> {
    for name in aliases {
        if weights.contains_tensor(name) {
            return weights.tensor(name);
        }
    }
    weights.tensor(aliases.first().map_or("missing_tensor", String::as_str))
}

pub(crate) fn is_2d_shape_compatible(shape: &[usize], in_dim: usize, out_dim: usize) -> bool {
    shape.len() != 2
        || (shape[0] == in_dim && shape[1] == out_dim)
        || (shape[0] == out_dim && shape[1] == in_dim)
}

pub(crate) fn validate_1d_view(
    view: &TensorView<'_>,
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let name = view.name();
    if view.num_elements() != expected_len {
        return Err(bounds_err(
            name,
            expected_len,
            view.num_elements(),
            format!(
                "tensor '{name}' expected {expected_len} elements, got {}",
                view.num_elements()
            ),
        ));
    }
    dequantize_all_finite(view, name)
}

pub(crate) fn require_tensor_1d_aliases(
    weights: &dyn ModelWeights,
    aliases: &[&str],
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = lookup_tensor_aliases(weights, aliases)?;
    validate_1d_view(&view, expected_len)
}

pub(crate) fn require_tensor_1d_owned(
    weights: &dyn ModelWeights,
    aliases: &[String],
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = lookup_tensor_owned_aliases(weights, aliases)?;
    validate_1d_view(&view, expected_len)
}

pub(crate) fn require_tensor_2d_owned(
    weights: &dyn ModelWeights,
    aliases: &[String],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = lookup_tensor_owned_aliases(weights, aliases)?;
    let name = view.name();
    let expected = checked_mul(name, in_dim, out_dim)?;
    let shape = view.shape();
    if !is_2d_shape_compatible(shape, in_dim, out_dim) {
        return Err(bounds_err(
            name,
            expected,
            view.num_elements(),
            format!("2D tensor '{name}' expected shape [{in_dim}, {out_dim}], got {shape:?}"),
        ));
    }
    if view.num_elements() != expected {
        return Err(bounds_err(
            name,
            expected,
            view.num_elements(),
            format!(
                "tensor '{name}' expected {expected} elements, got {}",
                view.num_elements()
            ),
        ));
    }
    dequantize_all_finite(&view, name)
}

pub(crate) fn validate_matvec_view(
    view: TensorView<'_>,
    in_dim: usize,
    out_dim: usize,
) -> Result<TensorView<'_>, K3Error> {
    let name = view.name();
    let expected = checked_mul(name, in_dim, out_dim)?;
    let shape = view.shape();
    if !is_2d_shape_compatible(shape, in_dim, out_dim) {
        return Err(bounds_err(
            name,
            expected,
            view.num_elements(),
            format!("2D tensor '{name}' expected shape [{in_dim}, {out_dim}], got {shape:?}"),
        ));
    }
    if view.num_elements() != expected {
        return Err(bounds_err(
            name,
            expected,
            view.num_elements(),
            format!(
                "tensor '{name}' expected {expected} elements, got {}",
                view.num_elements()
            ),
        ));
    }
    validate_raw_unquantized_finite(&view, name)?;
    Ok(view)
}

pub(crate) fn require_matvec_view_aliases<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[&str],
    in_dim: usize,
    out_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let view = lookup_tensor_aliases(weights, aliases)?;
    validate_matvec_view(view, in_dim, out_dim)
}

pub(crate) fn require_matvec_view_owned<'w>(
    weights: &'w dyn ModelWeights,
    aliases: &[String],
    in_dim: usize,
    out_dim: usize,
) -> Result<TensorView<'w>, K3Error> {
    let view = lookup_tensor_owned_aliases(weights, aliases)?;
    validate_matvec_view(view, in_dim, out_dim)
}

pub(crate) fn lookup_optional_bias_aliases(
    weights: &dyn ModelWeights,
    aliases: &[&str],
    expected_len: usize,
) -> Result<Option<Vec<f32>>, K3Error> {
    if contains_any_tensor(weights, aliases) {
        Ok(Some(require_tensor_1d_aliases(
            weights,
            aliases,
            expected_len,
        )?))
    } else {
        Ok(None)
    }
}

pub(crate) fn project_mat_view(
    view: &TensorView<'_>,
    input: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, K3Error> {
    let mut out = vec![0.0_f32; out_dim];
    matvec_any(&mut out, view, input)?;
    if let Some(b) = bias {
        if b.len() != out_dim {
            return Err(bounds_err(
                view.name(),
                out_dim,
                b.len(),
                format!(
                    "bias length {} does not match projection out_dim {out_dim}",
                    b.len()
                ),
            ));
        }
        for (dst, &bv) in out.iter_mut().zip(b.iter()) {
            *dst += bv;
        }
    }
    validate_finite_slice(view.name(), &out)?;
    Ok(out)
}

pub(crate) fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

pub(crate) fn softmax_inplace(scores: &mut [f32]) -> Result<(), K3Error> {
    if scores.is_empty() {
        return Ok(());
    }
    validate_finite_slice("attention_scores", scores)?;
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for s in scores.iter_mut() {
        let exp = (*s - max_val).exp();
        *s = exp;
        sum += exp;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(bounds_err(
            "attention_scores",
            scores.len(),
            scores.len(),
            "softmax normalizer became non-finite or non-positive".to_owned(),
        ));
    }
    let inv = 1.0_f32 / sum;
    for s in scores.iter_mut() {
        *s *= inv;
    }
    Ok(())
}

pub(crate) fn run_bidirectional_mha(
    q_seq: &[f32],
    k_seq: &[f32],
    v_seq: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    let d_model = checked_mul("mha_d_model", n_heads, head_dim)?;
    #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; seq_len * d_model];
    let mut scores = vec![0.0_f32; seq_len];

    for q_pos in 0..seq_len {
        let q_row = &q_seq[q_pos * d_model..(q_pos + 1) * d_model];
        let out_row = &mut out[q_pos * d_model..(q_pos + 1) * d_model];
        for h in 0..n_heads {
            let q_head = &q_row[h * head_dim..(h + 1) * head_dim];
            for k_pos in 0..seq_len {
                let k_head =
                    &k_seq[k_pos * d_model + h * head_dim..k_pos * d_model + (h + 1) * head_dim];
                scores[k_pos] = dot_slice(q_head, k_head) * scale;
            }
            softmax_inplace(&mut scores)?;
            let out_head = &mut out_row[h * head_dim..(h + 1) * head_dim];
            for k_pos in 0..seq_len {
                let weight = scores[k_pos];
                let v_head =
                    &v_seq[k_pos * d_model + h * head_dim..k_pos * d_model + (h + 1) * head_dim];
                for (dst, &vv) in out_head.iter_mut().zip(v_head.iter()) {
                    *dst += weight * vv;
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn infer_paired_2d_dim(
    shape: &[usize],
    known_dim: usize,
    gguf_is_dim0: bool,
) -> Option<usize> {
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return None;
    }
    if known_dim > 0 {
        if shape[0] == known_dim {
            return Some(shape[1]);
        }
        if shape[1] == known_dim {
            return Some(shape[0]);
        }
    }
    Some(if gguf_is_dim0 { shape[0] } else { shape[1] })
}

pub(crate) fn infer_layer_count_from_metadata(
    metadata: &ModelMetadata,
    prefixes: &[&str],
) -> usize {
    if metadata.layer_count() > 0 {
        return metadata.layer_count();
    }
    let mut max_idx: Option<usize> = None;
    for desc in metadata.tensors() {
        for prefix in prefixes {
            if let Some(rest) = desc.name().strip_prefix(prefix)
                && let Some((idx_str, _)) = rest.split_once('.')
                && let Ok(idx) = idx_str.parse::<usize>()
            {
                max_idx = Some(max_idx.map_or(idx, |cur| cur.max(idx)));
            }
        }
    }
    max_idx.map_or(0, |idx| idx.saturating_add(1))
}

pub(crate) fn parse_st_usize(
    st: &SafeTensorsFile<'_>,
    keys: &[&str],
    min_val: usize,
) -> Option<usize> {
    for &key in keys {
        if let Some(raw) = st.metadata_value(key)
            && let Ok(val) = raw.trim().parse::<usize>()
            && val >= min_val
        {
            return Some(val);
        }
    }
    None
}

pub(crate) fn parse_st_f32(st: &SafeTensorsFile<'_>, keys: &[&str]) -> Option<f32> {
    for &key in keys {
        if let Some(raw) = st.metadata_value(key)
            && let Ok(val) = raw.trim().parse::<f32>()
            && val.is_finite()
            && val > 0.0
        {
            return Some(val);
        }
    }
    None
}

pub(crate) fn validate_quant_and_arch(
    recipe: &dyn ArchRecipe,
    metadata: &ModelMetadata,
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let arch = metadata.architecture();
    if arch != "unknown" && arch != recipe.architecture_id() {
        return Err(K3Error::CapabilityMismatch {
            architecture: arch.to_owned(),
            reason: format!(
                "model architecture '{arch}' does not match recipe '{}'",
                recipe.architecture_id()
            ),
        });
    }
    for desc in metadata.tensors() {
        if !recipe.supports_quant(desc.quant()) {
            return Err(K3Error::unsupported_tensor_quant(
                arch,
                desc.name(),
                desc.quant(),
            ));
        }
        if weights.tensor_count() > 0
            && let Ok(view) = weights.tensor(desc.name())
            && !recipe.supports_quant(view.quant())
        {
            return Err(K3Error::unsupported_tensor_quant(
                arch,
                view.name(),
                view.quant(),
            ));
        }
    }
    Ok(())
}
