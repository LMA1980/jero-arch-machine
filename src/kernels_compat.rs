//! Compute kernel dispatch across `with-kernels` (`jero-kernels-rvv`) and portable scalar builds.

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
pub(crate) use jero_kernels_rvv::{
    AxisMlpWeights, BaselineKernel, TabPfnAttentionAxis, TabPfnAttentionShape, VectorKernel,
    dequantize_token_row, gelu_inplace_f32, gqa_causal_attention_f32, layernorm_f32, swiglu_f32,
    tabpfn_dual_axis_attention_f32, tokenize_continuous_features_f32,
    ttm_time_channel_mix_scratch_len, ttm_time_channel_mix_with_scratch_f32, vec_dot_bf16,
    vec_dot_f16, vec_dot_i2_s, vec_dot_iq4_nl, vec_dot_q2_k, vec_dot_q3_k, vec_dot_q4_0,
    vec_dot_q4_1, vec_dot_q5_0, vec_dot_q5_1, vec_dot_q5_k, vec_dot_q6_k, vec_dot_q8_0,
};

#[cfg(not(feature = "with-kernels"))]
use crate::common::{
    bounds_err, checked_mul, mismatch_err, softmax_inplace, validate_finite_slice,
};

/// Returns `true` if `quant` is supported by the active kernel configuration.
pub(crate) const fn is_supported_quant(quant: QuantType) -> bool {
    #[cfg(feature = "with-kernels")]
    {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q4_1
                | QuantType::Q5_0
                | QuantType::Q5_1
                | QuantType::Q2_K
                | QuantType::Q3_K
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
                | QuantType::Q8_0
                | QuantType::IQ4_NL
        )
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16)
    }
}

pub(crate) fn rmsnorm_standard(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().rmsnorm(output, input, weight, eps)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        rmsnorm_scalar(output, input, weight, eps)
    }
}

pub(crate) fn apply_rope_inplace(
    qk: &mut [f32],
    head_dim: usize,
    pos: usize,
    seq_len: usize,
    theta_base: f32,
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        jero_kernels_rvv::apply_rope_inplace(qk, head_dim, pos, seq_len, theta_base, None)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        apply_rope_neox_scalar(qk, head_dim, pos, seq_len, theta_base)
    }
}

/// Dispatches matrix-vector multiplication `out = weights * input` across all supported formats.
pub(crate) fn matvec_any(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    if out.is_empty() || input.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "matvec output and input slices cannot be empty".to_owned(),
        });
    }
    #[cfg(feature = "with-kernels")]
    {
        match weights.quant() {
            QuantType::F32 => matvec_f32(out, weights, input),
            QuantType::F16 => vec_dot_f16(out, weights, input),
            QuantType::BF16 => vec_dot_bf16(out, weights, input),
            QuantType::Q4_0 => vec_dot_q4_0(out, weights, input),
            QuantType::Q4_1 => vec_dot_q4_1(out, weights, input),
            QuantType::Q5_0 => vec_dot_q5_0(out, weights, input),
            QuantType::Q5_1 => vec_dot_q5_1(out, weights, input),
            QuantType::Q8_0 => vec_dot_q8_0(out, weights, input),
            QuantType::Q2_K => vec_dot_q2_k(out, weights, input),
            QuantType::Q3_K => vec_dot_q3_k(out, weights, input),
            QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(out, weights, input),
            QuantType::Q5_K => vec_dot_q5_k(out, weights, input),
            QuantType::Q6_K => vec_dot_q6_k(out, weights, input),
            QuantType::IQ4_NL => vec_dot_iq4_nl(out, weights, input),
            QuantType::I2_S => vec_dot_i2_s(out, weights, input),
            other => Err(K3Error::unsupported_tensor_quant(
                "machine",
                weights.name(),
                other,
            )),
        }
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        match weights.quant() {
            QuantType::F32 => matvec_f32(out, weights, input),
            QuantType::F16 | QuantType::BF16 => matvec_half_scalar(out, weights, input),
            other => Err(K3Error::unsupported_tensor_quant(
                "machine",
                weights.name(),
                other,
            )),
        }
    }
}

fn matvec_f32(out: &mut [f32], weights: &TensorView<'_>, input: &[f32]) -> Result<(), K3Error> {
    let rows = out.len();
    let cols = input.len();
    let expected_elems = rows
        .checked_mul(cols)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "F32 matvec dimension overflowed usize".to_owned(),
        })?;
    if weights.num_elements() != expected_elems {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "F32 tensor elements {} != out({rows}) * in({cols})",
                weights.num_elements()
            ),
        });
    }
    let raw = weights.as_bytes();
    let required_bytes = expected_elems * size_of::<f32>();
    if raw.len() < required_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: raw.len(),
            reason: format!(
                "F32 buffer length {} < required {required_bytes}",
                raw.len()
            ),
        });
    }

    #[cfg(feature = "with-kernels")]
    let kernel = BaselineKernel::new();
    let mut row_buf = vec![0.0_f32; cols];
    for (r, out_slot) in out.iter_mut().enumerate() {
        let base = r * cols * size_of::<f32>();
        for (c, slot) in row_buf.iter_mut().enumerate() {
            let off = base + c * size_of::<f32>();
            *slot = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
        }
        #[cfg(feature = "with-kernels")]
        {
            *out_slot = kernel.dot_product_f32(&row_buf, input);
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            *out_slot = row_buf.iter().zip(input.iter()).map(|(&a, &b)| a * b).sum();
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn decode_half_bits(quant: QuantType, bits: u16) -> f32 {
    match quant {
        QuantType::F16 => {
            let v = jero_core::f16_bits_to_f32(bits);
            if v.is_finite() { v } else { 0.0_f32 }
        }
        QuantType::BF16 => {
            let exp = (bits >> 7) & 0xFF;
            if exp == 0xFF {
                0.0_f32
            } else {
                f32::from_bits(u32::from(bits) << 16)
            }
        }
        _ => 0.0_f32,
    }
}

#[cfg(not(feature = "with-kernels"))]
fn matvec_half_scalar(
    out: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let rows = out.len();
    let cols = input.len();
    let expected = checked_mul(weights.name(), rows, cols)?;
    if weights.num_elements() != expected {
        return Err(bounds_err(
            weights.name(),
            expected,
            weights.num_elements(),
            format!(
                "half-precision matrix elements {} != out({rows}) * in({cols})",
                weights.num_elements()
            ),
        ));
    }
    let raw = weights.as_bytes();
    let req_bytes = checked_mul(weights.name(), expected, 2)?;
    if raw.len() < req_bytes {
        return Err(bounds_err(
            weights.name(),
            expected,
            raw.len() / 2,
            format!("half-precision buffer length {} < {req_bytes}", raw.len()),
        ));
    }
    let quant = weights.quant();
    for (r, dst) in out.iter_mut().enumerate() {
        let base = r * cols * 2;
        let mut acc = 0.0_f32;
        for (c, &x) in input.iter().enumerate() {
            let off = base + c * 2;
            let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
            acc += decode_half_bits(quant, bits) * x;
        }
        *dst = acc;
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
pub(crate) fn dequantize_token_row(
    table: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    let cols = out.len();
    if cols == 0 {
        return Err(bounds_err(
            table.name(),
            1,
            0,
            "empty output row".to_owned(),
        ));
    }
    let total = table.num_elements();
    if total == 0 || total % cols != 0 {
        return Err(bounds_err(
            table.name(),
            cols,
            total,
            format!("element count {total} not divisible by row width {cols}"),
        ));
    }
    let num_rows = total / cols;
    if row_idx >= num_rows {
        return Err(bounds_err(
            table.name(),
            row_idx + 1,
            num_rows,
            format!("row index {row_idx} out of bounds for {num_rows} rows"),
        ));
    }
    let raw = table.as_bytes();
    match table.quant() {
        QuantType::F32 => {
            let row_bytes = cols * 4;
            let start = row_idx * row_bytes;
            if raw.len() < start + row_bytes {
                return Err(bounds_err(
                    table.name(),
                    total,
                    raw.len() / 4,
                    "truncated F32 tensor".to_owned(),
                ));
            }
            for (c, dst) in out.iter_mut().enumerate() {
                let off = start + c * 4;
                *dst = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            }
            Ok(())
        }
        q @ (QuantType::F16 | QuantType::BF16) => {
            let row_bytes = cols * 2;
            let start = row_idx * row_bytes;
            if raw.len() < start + row_bytes {
                return Err(bounds_err(
                    table.name(),
                    total,
                    raw.len() / 2,
                    "truncated half-precision tensor".to_owned(),
                ));
            }
            for (c, dst) in out.iter_mut().enumerate() {
                let off = start + c * 2;
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                *dst = decode_half_bits(q, bits);
            }
            Ok(())
        }
        other => Err(K3Error::unsupported_tensor_quant(
            "machine",
            table.name(),
            other,
        )),
    }
}

#[cfg(not(feature = "with-kernels"))]
pub(crate) fn gelu_inplace_f32(values: &mut [f32]) -> Result<(), K3Error> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const CUBIC: f32 = 0.044_715;
    if values.is_empty() {
        return Err(bounds_err(
            "gelu_values",
            1,
            0,
            "gelu_values cannot be empty".to_owned(),
        ));
    }
    validate_finite_slice("gelu_values", values)?;
    for x in values.iter_mut() {
        let v = *x;
        let inner = SQRT_2_OVER_PI * (v + CUBIC * v * v * v);
        *x = 0.5_f32 * v * (1.0_f32 + inner.tanh());
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
pub(crate) fn layernorm_f32(
    out: &mut [f32],
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), K3Error> {
    let len = input.len();
    if len == 0 || out.len() != len || !eps.is_finite() || eps <= 0.0 {
        return Err(bounds_err(
            "layernorm",
            len.max(1),
            out.len(),
            "invalid layernorm geometry or eps".to_owned(),
        ));
    }
    validate_finite_slice("layernorm_input", input)?;
    if let Some(w) = weight {
        if w.len() != len {
            return Err(bounds_err(
                "layernorm_weight",
                len,
                w.len(),
                "layernorm weight length mismatch".to_owned(),
            ));
        }
        validate_finite_slice("layernorm_weight", w)?;
    }
    if let Some(b) = bias {
        if b.len() != len {
            return Err(bounds_err(
                "layernorm_bias",
                len,
                b.len(),
                "layernorm bias length mismatch".to_owned(),
            ));
        }
        validate_finite_slice("layernorm_bias", b)?;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "vector length fits cleanly in f32"
    )]
    let inv_n = 1.0_f32 / (len as f32);
    let mean = input.iter().copied().sum::<f32>() * inv_n;
    let var = input
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f32>()
        * inv_n;
    let inv_std = 1.0_f32 / (var + eps).sqrt();
    for (i, &x) in input.iter().enumerate() {
        let mut y = (x - mean) * inv_std;
        if let Some(w) = weight {
            y *= w[i];
        }
        if let Some(b) = bias {
            y += b[i];
        }
        out[i] = y;
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
pub(crate) fn tokenize_continuous_features_f32(
    out: &mut [f32],
    raw: &[f32],
    weight: &[f32],
    bias: &[f32],
    n_features: usize,
    d_token: usize,
) -> Result<(), K3Error> {
    let expected = checked_mul("tokenize_continuous", n_features, d_token)?;
    if raw.len() != n_features
        || weight.len() != expected
        || bias.len() != expected
        || out.len() != expected
    {
        return Err(bounds_err(
            "tokenize_continuous",
            expected,
            out.len(),
            "continuous tokenizer slice length mismatch".to_owned(),
        ));
    }
    validate_finite_slice("tokenize_raw", raw)?;
    validate_finite_slice("tokenize_weight", weight)?;
    validate_finite_slice("tokenize_bias", bias)?;
    for (i, &x) in raw.iter().enumerate() {
        let base = i * d_token;
        for d in 0..d_token {
            out[base + d] = x.mul_add(weight[base + d], bias[base + d]);
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct AxisMlpWeights<'a> {
    pub(crate) norm_weight: &'a [f32],
    pub(crate) norm_bias: &'a [f32],
    pub(crate) fc1_weight: &'a [f32],
    pub(crate) fc1_bias: &'a [f32],
    pub(crate) fc2_weight: &'a [f32],
    pub(crate) fc2_bias: &'a [f32],
    pub(crate) dim: usize,
    pub(crate) hidden_dim: usize,
}

#[cfg(not(feature = "with-kernels"))]
#[must_use]
pub(crate) fn ttm_time_channel_mix_scratch_len(
    n_patches: usize,
    n_channels: usize,
    time_hidden_dim: usize,
    channel_hidden_dim: usize,
) -> Option<usize> {
    if n_patches == 0 || n_channels == 0 || time_hidden_dim == 0 || channel_hidden_dim == 0 {
        return None;
    }
    let stage = n_patches.checked_mul(n_channels)?;
    let time_work = n_patches.checked_mul(3)?.checked_add(time_hidden_dim)?;
    let chan_work = n_channels.checked_add(channel_hidden_dim)?;
    stage.checked_add(time_work.max(chan_work))
}

#[cfg(not(feature = "with-kernels"))]
fn run_axis_mlp_scalar(
    norm_buf: &mut [f32],
    hidden_buf: &mut [f32],
    out_vec: &mut [f32],
    x_vec: &[f32],
    weights: &AxisMlpWeights<'_>,
    eps: f32,
) -> Result<(), K3Error> {
    let d = weights.dim;
    let h_dim = weights.hidden_dim;
    layernorm_f32(
        &mut norm_buf[..d],
        x_vec,
        Some(weights.norm_weight),
        Some(weights.norm_bias),
        eps,
    )?;
    for (h, &b1) in weights.fc1_bias.iter().enumerate() {
        let row = &weights.fc1_weight[h * d..(h + 1) * d];
        let dot: f32 = row
            .iter()
            .zip(norm_buf[..d].iter())
            .map(|(&a, &b)| a * b)
            .sum();
        hidden_buf[h] = dot + b1;
    }
    gelu_inplace_f32(&mut hidden_buf[..h_dim])?;
    for (idx, (&xv, &b2)) in x_vec.iter().zip(weights.fc2_bias.iter()).enumerate() {
        let row = &weights.fc2_weight[idx * h_dim..(idx + 1) * h_dim];
        let dot: f32 = row
            .iter()
            .zip(hidden_buf[..h_dim].iter())
            .map(|(&a, &b)| a * b)
            .sum();
        out_vec[idx] = xv + dot + b2;
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "matches jero-kernels-rvv ttm_time_channel_mix_with_scratch_f32 signature"
)]
pub(crate) fn ttm_time_channel_mix_with_scratch_f32(
    out: &mut [f32],
    input: &[f32],
    n_patches: usize,
    n_channels: usize,
    time_weights: &AxisMlpWeights<'_>,
    channel_weights: &AxisMlpWeights<'_>,
    eps: f32,
    scratch: &mut [f32],
) -> Result<(), K3Error> {
    let total = checked_mul("ttm_mix", n_patches, n_channels)?;
    let req = ttm_time_channel_mix_scratch_len(
        n_patches,
        n_channels,
        time_weights.hidden_dim,
        channel_weights.hidden_dim,
    )
    .ok_or_else(|| bounds_err("ttm_scratch", 0, 0, "ttm scratch overflow".to_owned()))?;
    if out.len() != total || input.len() != total || scratch.len() < req {
        return Err(bounds_err(
            "ttm_mix",
            total,
            out.len(),
            "TTM mixer buffer or scratch mismatch".to_owned(),
        ));
    }
    let (stage1, work) = scratch.split_at_mut(total);
    {
        let (x_col, rem1) = work.split_at_mut(n_patches);
        let (out_col, rem2) = rem1.split_at_mut(n_patches);
        let (norm_buf, hidden_buf) = rem2.split_at_mut(n_patches);
        for c in 0..n_channels {
            for p in 0..n_patches {
                x_col[p] = input[p * n_channels + c];
            }
            run_axis_mlp_scalar(
                norm_buf,
                &mut hidden_buf[..time_weights.hidden_dim],
                out_col,
                x_col,
                time_weights,
                eps,
            )?;
            for p in 0..n_patches {
                stage1[p * n_channels + c] = out_col[p];
            }
        }
    }
    let (norm_c, hidden_c) = work.split_at_mut(n_channels);
    for p in 0..n_patches {
        let base = p * n_channels;
        run_axis_mlp_scalar(
            norm_c,
            &mut hidden_c[..channel_weights.hidden_dim],
            &mut out[base..base + n_channels],
            &stage1[base..base + n_channels],
            channel_weights,
            eps,
        )?;
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TabPfnAttentionAxis {
    Feature,
    Sample,
}

#[cfg(not(feature = "with-kernels"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TabPfnAttentionShape {
    pub(crate) n_samples: usize,
    pub(crate) n_features: usize,
    pub(crate) n_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) context_samples: Option<usize>,
}

#[cfg(not(feature = "with-kernels"))]
impl TabPfnAttentionShape {
    pub(crate) const fn new(
        n_samples: usize,
        n_features: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            n_samples,
            n_features,
            n_heads,
            head_dim,
            context_samples: None,
        }
    }

    pub(crate) const fn with_context_samples(mut self, context_samples: usize) -> Self {
        self.context_samples = Some(context_samples);
        self
    }

    pub(crate) const fn total_elements(self) -> Option<usize> {
        let Some(d) = self.n_heads.checked_mul(self.head_dim) else {
            return None;
        };
        let Some(row) = self.n_features.checked_mul(d) else {
            return None;
        };
        self.n_samples.checked_mul(row)
    }
}

#[cfg(not(feature = "with-kernels"))]
pub(crate) fn tabpfn_dual_axis_attention_f32(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: TabPfnAttentionShape,
    axis: TabPfnAttentionAxis,
    scale: f32,
) -> Result<(), K3Error> {
    let total = shape
        .total_elements()
        .ok_or_else(|| bounds_err("tabpfn_shape", 0, 0, "shape overflow".to_owned()))?;
    let d_model = shape.n_heads * shape.head_dim;
    let n_ctx = shape.context_samples.unwrap_or(shape.n_samples);
    if total == 0
        || n_ctx == 0
        || n_ctx > shape.n_samples
        || out.len() != total
        || q.len() != total
        || k.len() != total
        || v.len() != total
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(bounds_err(
            "tabpfn_attention",
            total,
            out.len(),
            "invalid TabPFN attention shape or buffers".to_owned(),
        ));
    }
    let (outer_len, q_len, kv_len, outer_step, pos_step) = match axis {
        TabPfnAttentionAxis::Feature => (
            shape.n_samples,
            shape.n_features,
            shape.n_features,
            shape.n_features * d_model,
            d_model,
        ),
        TabPfnAttentionAxis::Sample => (
            shape.n_features,
            shape.n_samples,
            n_ctx,
            d_model,
            shape.n_features * d_model,
        ),
    };
    let mut scores = vec![0.0_f32; kv_len];
    for outer in 0..outer_len {
        let outer_base = outer * outer_step;
        for h in 0..shape.n_heads {
            let head_base = outer_base + h * shape.head_dim;
            for qi in 0..q_len {
                let q_off = head_base + qi * pos_step;
                let q_head = &q[q_off..q_off + shape.head_dim];
                for (kj, slot) in scores.iter_mut().enumerate() {
                    let k_off = head_base + kj * pos_step;
                    let dot: f32 = q_head
                        .iter()
                        .zip(k[k_off..k_off + shape.head_dim].iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    *slot = dot * scale;
                }
                softmax_inplace(&mut scores)?;
                let out_head = &mut out[q_off..q_off + shape.head_dim];
                out_head.fill(0.0);
                for (kj, &w) in scores.iter().enumerate() {
                    let v_off = head_base + kj * pos_step;
                    for (dst, &vv) in out_head
                        .iter_mut()
                        .zip(v[v_off..v_off + shape.head_dim].iter())
                    {
                        *dst += w * vv;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn rmsnorm_scalar(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let n = input.len();
    if n == 0 || output.len() != n || weight.len() != n || !eps.is_finite() || eps <= 0.0 {
        return Err(bounds_err(
            "rmsnorm",
            n.max(1),
            output.len(),
            "invalid rmsnorm slice dimensions or eps".to_owned(),
        ));
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "vector length fits cleanly in f32"
    )]
    let mean_sq = input.iter().map(|&x| x * x).sum::<f32>() / (n as f32);
    let inv_rms = 1.0_f32 / (mean_sq + eps).sqrt();
    for ((dst, &x), &w) in output.iter_mut().zip(input.iter()).zip(weight.iter()) {
        *dst = x * inv_rms * w;
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn apply_rope_neox_scalar(
    qk: &mut [f32],
    head_dim: usize,
    pos: usize,
    _seq_len: usize,
    theta_base: f32,
) -> Result<(), K3Error> {
    if head_dim < 2 || head_dim % 2 != 0 || qk.is_empty() || qk.len() % head_dim != 0 {
        return Err(bounds_err(
            "rope",
            head_dim,
            qk.len(),
            "invalid RoPE dimensions".to_owned(),
        ));
    }
    if !theta_base.is_finite() || theta_base <= 0.0 {
        return Err(mismatch_err("rope", "theta_base must be finite and > 0"));
    }
    let half = head_dim / 2;
    #[expect(clippy::cast_precision_loss, reason = "head_dim and pos fit in f32")]
    let pos_f = pos as f32;
    #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
    let dim_f = head_dim as f32;
    for head in qk.chunks_exact_mut(head_dim) {
        for i in 0..half {
            #[expect(clippy::cast_precision_loss, reason = "pair index fits in f32")]
            let exp = (2 * i) as f32 / dim_f;
            let angle = pos_f / theta_base.powf(exp);
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos_a - x1 * sin_a;
            head[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
#[expect(
    clippy::too_many_arguments,
    reason = "matches jero-kernels-rvv gqa_causal_attention_f32 signature"
)]
pub(crate) fn gqa_causal_attention_f32(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Result<(), K3Error> {
    if n_heads_q == 0
        || n_heads_kv == 0
        || head_dim == 0
        || seq_len == 0
        || n_heads_q % n_heads_kv != 0
    {
        return Err(mismatch_err("gqa", "invalid GQA head/sequence dimensions"));
    }
    let q_dim = n_heads_q * head_dim;
    let kv_dim = n_heads_kv * head_dim;
    if out.len() != q_dim
        || q.len() != q_dim
        || k_cache.len() != seq_len * kv_dim
        || v_cache.len() != seq_len * kv_dim
    {
        return Err(bounds_err(
            "gqa",
            q_dim,
            out.len(),
            "GQA buffer length mismatch".to_owned(),
        ));
    }
    let group_size = n_heads_q / n_heads_kv;
    let mut scores = vec![0.0_f32; seq_len];
    for h_q in 0..n_heads_q {
        let h_kv = h_q / group_size;
        let q_head = &q[h_q * head_dim..(h_q + 1) * head_dim];
        for (t, slot) in scores.iter_mut().enumerate() {
            let k_off = t * kv_dim + h_kv * head_dim;
            let dot: f32 = q_head
                .iter()
                .zip(k_cache[k_off..k_off + head_dim].iter())
                .map(|(&a, &b)| a * b)
                .sum();
            *slot = dot * scale;
        }
        softmax_inplace(&mut scores)?;
        let out_head = &mut out[h_q * head_dim..(h_q + 1) * head_dim];
        out_head.fill(0.0);
        for (t, &w) in scores.iter().enumerate() {
            let v_off = t * kv_dim + h_kv * head_dim;
            for (dst, &vv) in out_head
                .iter_mut()
                .zip(v_cache[v_off..v_off + head_dim].iter())
            {
                *dst += w * vv;
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
pub(crate) fn swiglu_f32(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    if gate.is_empty() || out.len() != gate.len() || up.len() != gate.len() {
        return Err(bounds_err(
            "swiglu",
            gate.len().max(1),
            out.len(),
            "swiglu slice lengths mismatch".to_owned(),
        ));
    }
    for ((dst, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        let silu = if g < -20.0 {
            0.0_f32
        } else if g > 20.0 {
            g
        } else {
            g / (1.0_f32 + (-g).exp())
        };
        *dst = silu * u;
    }
    Ok(())
}
