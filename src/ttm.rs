//! Zero-KV `TinyTimeMixers` (`TTM`) multivariate zero-shot time-series forecasting recipe.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelMetadata, ModelWeights,
    QuantType, SafeTensorsFile,
};

use crate::common::{
    bounds_err, checked_mul, contains_any_tensor, find_tensor_by_aliases,
    infer_layer_count_from_metadata, infer_paired_2d_dim, lookup_optional_bias_aliases,
    mismatch_err, parse_st_f32, parse_st_usize, project_mat_view, require_matvec_view_aliases,
    require_tensor_1d_aliases, require_tensor_1d_owned, require_tensor_2d_owned,
    validate_finite_slice, validate_quant_and_arch,
};
use crate::kernels_compat::{
    AxisMlpWeights, is_supported_quant, ttm_time_channel_mix_scratch_len,
    ttm_time_channel_mix_with_scratch_f32,
};

const PATCH_PROJ_WEIGHT_ALIASES: &[&str] = &[
    "ttm.patch_proj.weight",
    "backbone.patcher.weight",
    "patcher.weight",
    "encoder.patcher.weight",
];
const PATCH_PROJ_BIAS_ALIASES: &[&str] = &[
    "ttm.patch_proj.bias",
    "backbone.patcher.bias",
    "patcher.bias",
    "encoder.patcher.bias",
];
const FORECAST_HEAD_WEIGHT_ALIASES: &[&str] = &[
    "ttm.forecast_head.weight",
    "head.projection.weight",
    "head.weight",
    "forecast_head.weight",
];
const FORECAST_HEAD_BIAS_ALIASES: &[&str] = &[
    "ttm.forecast_head.bias",
    "head.projection.bias",
    "head.bias",
    "forecast_head.bias",
];
const LAYER0_TIME_NORM_ALIASES: &[&str] = &[
    "blk.0.time_norm.weight",
    "backbone.encoder.layers.0.patch_mixer.norm.weight",
    "encoder.layers.0.time_norm.weight",
    "layers.0.time_norm.weight",
];
const LAYER0_CHANNEL_NORM_ALIASES: &[&str] = &[
    "blk.0.channel_norm.weight",
    "backbone.encoder.layers.0.channel_mixer.norm.weight",
    "encoder.layers.0.channel_norm.weight",
    "layers.0.channel_norm.weight",
];
const LAYER0_TIME_FC1_BIAS_ALIASES: &[&str] = &[
    "blk.0.time_mlp.fc1.bias",
    "backbone.encoder.layers.0.patch_mixer.mlp.fc1.bias",
    "encoder.layers.0.time_mlp.fc1.bias",
    "layers.0.time_mlp.fc1.bias",
];
const LAYER0_CHAN_FC1_BIAS_ALIASES: &[&str] = &[
    "blk.0.channel_mlp.fc1.bias",
    "backbone.encoder.layers.0.channel_mixer.mlp.fc1.bias",
    "encoder.layers.0.channel_mlp.fc1.bias",
    "layers.0.channel_mlp.fc1.bias",
];
const LAYER_PREFIXES: &[&str] = &[
    "blk.",
    "backbone.encoder.layers.",
    "encoder.layers.",
    "layers.",
];

fn layer_aliases(layer: usize, gguf_suffix: &str, hf_suffix: &str) -> [String; 4] {
    [
        format!("blk.{layer}.{gguf_suffix}"),
        format!("backbone.encoder.layers.{layer}.{hf_suffix}"),
        format!("encoder.layers.{layer}.{gguf_suffix}"),
        format!("layers.{layer}.{gguf_suffix}"),
    ]
}

/// Zero-KV `TinyTimeMixers` (`TTM`) patch-time and channel-mixing time-series forecasting recipe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TtmRecipe {
    /// Input lookback series length (`L`, e.g., 64 or 512).
    pub context_length: usize,
    /// Patch window length (`P`, dividing `context_length`).
    pub patch_length: usize,
    /// Forecast horizon length (`F`, e.g., 16 or 96).
    pub forecast_length: usize,
    /// Number of multivariate telemetry channels (`C`).
    pub num_channels: usize,
    /// Number of stacked time+channel MLP mixer layers.
    pub num_mixer_layers: usize,
    /// Hidden width of the time/patch-axis MLP.
    pub time_hidden_dim: usize,
    /// Hidden width of the channel-axis MLP.
    pub channel_hidden_dim: usize,
    /// `LayerNorm` epsilon (`1e-5` default).
    pub norm_eps: f32,
}

impl Default for TtmRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl TtmRecipe {
    /// Creates a baseline `ttm` forecasting recipe sized conservatively (`n_ctx = 64`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            context_length: 64,
            patch_length: 8,
            forecast_length: 16,
            num_channels: 4,
            num_mixer_layers: 2,
            time_hidden_dim: 16,
            channel_hidden_dim: 8,
            norm_eps: 1e-5,
        }
    }

    /// Returns the number of temporal patches (`context_length / patch_length`).
    #[must_use]
    pub const fn num_patches(&self) -> usize {
        match self.context_length.checked_div(self.patch_length) {
            Some(v) => v,
            None => 0,
        }
    }

    /// Hydrates a [`TtmRecipe`] from [`ModelMetadata`] and GGUF or `SafeTensors` tensor descriptors.
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        if metadata.context_length() > 0 {
            recipe.context_length = metadata.context_length();
        }
        let layers = infer_layer_count_from_metadata(metadata, LAYER_PREFIXES);
        if layers > 0 {
            recipe.num_mixer_layers = layers;
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, LAYER0_CHANNEL_NORM_ALIASES)
            && desc.num_elements() > 0
        {
            recipe.num_channels = desc.num_elements();
        }
        let patches_opt = find_tensor_by_aliases(metadata, LAYER0_TIME_NORM_ALIASES)
            .map(jero_core::TensorDescriptor::num_elements)
            .filter(|&n| n > 0);
        let patch_proj_len = find_tensor_by_aliases(metadata, PATCH_PROJ_WEIGHT_ALIASES)
            .map(jero_core::TensorDescriptor::num_elements)
            .filter(|&n| n > 0);
        if let Some(patches) = patches_opt {
            if let Some(p_len) = patch_proj_len
                && let Some(prod) = patches.checked_mul(p_len)
                && (recipe.context_length == 2048 || recipe.context_length == prod)
            {
                recipe.patch_length = p_len;
                recipe.context_length = prod;
            } else if recipe.context_length % patches == 0 {
                recipe.patch_length = (recipe.context_length / patches).max(1);
            } else {
                recipe.patch_length = 1;
                recipe.context_length = patches;
            }
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, LAYER0_TIME_FC1_BIAS_ALIASES)
            && desc.num_elements() > 0
        {
            recipe.time_hidden_dim = desc.num_elements();
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, LAYER0_CHAN_FC1_BIAS_ALIASES)
            && desc.num_elements() > 0
        {
            recipe.channel_hidden_dim = desc.num_elements();
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, FORECAST_HEAD_BIAS_ALIASES)
            && desc.num_elements() > 0
        {
            recipe.forecast_length = desc.num_elements();
        } else if let Some(desc) = find_tensor_by_aliases(metadata, FORECAST_HEAD_WEIGHT_ALIASES)
            && let Some(fl) = infer_paired_2d_dim(desc.shape(), patches_opt.unwrap_or(0), false)
        {
            recipe.forecast_length = fl;
        }
        if let Some(eps) = metadata.rms_norm_eps()
            && eps.is_finite()
            && eps > 0.0
        {
            recipe.norm_eps = eps;
        }
        recipe
    }

    /// Hydrates a [`TtmRecipe`] from a parsed [`GgufFile`], reading optional `ttm.*` GGUF keys.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        for (key, dst) in [
            ("ttm.patch_length", &mut recipe.patch_length),
            ("ttm.forecast_length", &mut recipe.forecast_length),
            ("ttm.num_channels", &mut recipe.num_channels),
            ("ttm.time_hidden_dim", &mut recipe.time_hidden_dim),
            ("ttm.channel_hidden_dim", &mut recipe.channel_hidden_dim),
        ] {
            if let Some(v) = gguf.metadata_value(key).and_then(GgufValue::as_usize)
                && v > 0
            {
                *dst = v;
            }
        }
        recipe
    }

    /// Hydrates a [`TtmRecipe`] from a parsed [`SafeTensorsFile`], reading `"__metadata__"` keys
    /// and tensor descriptors uniformly.
    #[must_use]
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(st.metadata());
        if let Some(v) = parse_st_usize(st, &["ttm.context_length", "context_length"], 1) {
            recipe.context_length = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "ttm.block_count",
                "num_layers",
                "num_mixer_layers",
                "num_hidden_layers",
            ],
            1,
        ) {
            recipe.num_mixer_layers = v;
        }
        if let Some(v) = parse_st_usize(st, &["ttm.patch_length", "patch_length"], 1) {
            recipe.patch_length = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "ttm.forecast_length",
                "prediction_length",
                "forecast_length",
            ],
            1,
        ) {
            recipe.forecast_length = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &["ttm.num_channels", "num_input_channels", "num_channels"],
            1,
        ) {
            recipe.num_channels = v;
        }
        if let Some(v) = parse_st_usize(st, &["ttm.time_hidden_dim", "time_hidden_dim"], 1) {
            recipe.time_hidden_dim = v;
        }
        if let Some(v) = parse_st_usize(st, &["ttm.channel_hidden_dim", "channel_hidden_dim"], 1) {
            recipe.channel_hidden_dim = v;
        }
        if let Some(eps) = parse_st_f32(st, &["ttm.norm_eps", "layer_norm_eps", "norm_eps"]) {
            recipe.norm_eps = eps;
        }
        recipe
    }

    fn validate_geometry(&self) -> Result<usize, K3Error> {
        if self.context_length == 0
            || self.patch_length == 0
            || self.context_length % self.patch_length != 0
            || self.forecast_length == 0
            || self.num_channels == 0
            || self.num_mixer_layers == 0
            || self.time_hidden_dim == 0
            || self.channel_hidden_dim == 0
        {
            return Err(mismatch_err(
                "ttm",
                format!(
                    "invalid TTM geometry (ctx={}, patch={}, forecast={}, channels={}, layers={})",
                    self.context_length,
                    self.patch_length,
                    self.forecast_length,
                    self.num_channels,
                    self.num_mixer_layers
                ),
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(mismatch_err(
                "ttm",
                format!("ttm norm_eps ({}) must be finite and > 0", self.norm_eps),
            ));
        }
        Ok(self.context_length / self.patch_length)
    }

    fn validate_weights(&self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        let n_patches = self.validate_geometry()?;
        if self.patch_length > 1 {
            let _ =
                require_tensor_1d_aliases(weights, PATCH_PROJ_WEIGHT_ALIASES, self.patch_length)?;
        }
        for layer in 0..self.num_mixer_layers {
            let _ = TtmLayerTensors::load(
                weights,
                layer,
                n_patches,
                self.num_channels,
                self.time_hidden_dim,
                self.channel_hidden_dim,
            )?;
        }
        let _ = require_matvec_view_aliases(
            weights,
            FORECAST_HEAD_WEIGHT_ALIASES,
            n_patches,
            self.forecast_length,
        )?;
        let _ = lookup_optional_bias_aliases(
            weights,
            FORECAST_HEAD_BIAS_ALIASES,
            self.forecast_length,
        )?;
        Ok(())
    }

    fn prepare_patches(
        &self,
        features: &[f32],
        num_rows: usize,
        num_cols: usize,
        n_patches: usize,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        if num_cols != self.num_channels {
            return Err(bounds_err(
                "feature_input",
                self.num_channels,
                num_cols,
                format!(
                    "ttm expected {} channels (cols), got {num_cols}",
                    self.num_channels
                ),
            ));
        }
        if num_rows == n_patches
            && (self.patch_length == 1 || !contains_any_tensor(weights, PATCH_PROJ_WEIGHT_ALIASES))
        {
            return Ok(features.to_vec());
        }
        if num_rows != self.context_length {
            return Err(bounds_err(
                "feature_input",
                self.context_length * self.num_channels,
                features.len(),
                format!(
                    "ttm expected feature rows == context_length ({}) or n_patches ({n_patches}), got {num_rows}",
                    self.context_length
                ),
            ));
        }
        let proj_w =
            require_tensor_1d_aliases(weights, PATCH_PROJ_WEIGHT_ALIASES, self.patch_length)?;
        let proj_bias = lookup_optional_bias_aliases(weights, PATCH_PROJ_BIAS_ALIASES, 1)?;
        let bias = proj_bias.map_or(0.0_f32, |b| b[0]);
        let mut patched = vec![0.0_f32; n_patches * self.num_channels];
        for p in 0..n_patches {
            let t_base = p * self.patch_length;
            for c in 0..self.num_channels {
                let mut acc = bias;
                for (k, &w) in proj_w.iter().enumerate() {
                    acc += w * features[(t_base + k) * self.num_channels + c];
                }
                patched[p * self.num_channels + c] = acc;
            }
        }
        validate_finite_slice("ttm_patched", &patched)?;
        Ok(patched)
    }

    fn project_forecast_head(
        &self,
        weights: &dyn ModelWeights,
        state: &[f32],
        n_patches: usize,
        out_elems: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let head_view = require_matvec_view_aliases(
            weights,
            FORECAST_HEAD_WEIGHT_ALIASES,
            n_patches,
            self.forecast_length,
        )?;
        let head_bias = lookup_optional_bias_aliases(
            weights,
            FORECAST_HEAD_BIAS_ALIASES,
            self.forecast_length,
        )?;
        let mut chan_col = vec![0.0_f32; n_patches];
        let mut forecast = vec![0.0_f32; out_elems];
        for c in 0..self.num_channels {
            for p in 0..n_patches {
                chan_col[p] = state[p * self.num_channels + c];
            }
            let chan_out = project_mat_view(
                &head_view,
                &chan_col,
                self.forecast_length,
                head_bias.as_deref(),
            )?;
            for (h, &val) in chan_out.iter().enumerate() {
                forecast[h * self.num_channels + c] = val;
            }
        }
        Ok(forecast)
    }
}

struct TtmLayerTensors {
    time_norm_w: Vec<f32>,
    time_norm_b: Vec<f32>,
    time_fc1_w: Vec<f32>,
    time_fc1_b: Vec<f32>,
    time_fc2_w: Vec<f32>,
    time_fc2_b: Vec<f32>,
    chan_norm_w: Vec<f32>,
    chan_norm_b: Vec<f32>,
    chan_fc1_w: Vec<f32>,
    chan_fc1_b: Vec<f32>,
    chan_fc2_w: Vec<f32>,
    chan_fc2_b: Vec<f32>,
}

impl TtmLayerTensors {
    fn load(
        weights: &dyn ModelWeights,
        layer: usize,
        n_patches: usize,
        n_channels: usize,
        time_hidden: usize,
        chan_hidden: usize,
    ) -> Result<Self, K3Error> {
        Ok(Self {
            time_norm_w: require_tensor_1d_owned(
                weights,
                &layer_aliases(layer, "time_norm.weight", "patch_mixer.norm.weight"),
                n_patches,
            )?,
            time_norm_b: require_tensor_1d_owned(
                weights,
                &layer_aliases(layer, "time_norm.bias", "patch_mixer.norm.bias"),
                n_patches,
            )?,
            time_fc1_w: require_tensor_2d_owned(
                weights,
                &layer_aliases(layer, "time_mlp.fc1.weight", "patch_mixer.mlp.fc1.weight"),
                n_patches,
                time_hidden,
            )?,
            time_fc1_b: require_tensor_1d_owned(
                weights,
                &layer_aliases(layer, "time_mlp.fc1.bias", "patch_mixer.mlp.fc1.bias"),
                time_hidden,
            )?,
            time_fc2_w: require_tensor_2d_owned(
                weights,
                &layer_aliases(layer, "time_mlp.fc2.weight", "patch_mixer.mlp.fc2.weight"),
                time_hidden,
                n_patches,
            )?,
            time_fc2_b: require_tensor_1d_owned(
                weights,
                &layer_aliases(layer, "time_mlp.fc2.bias", "patch_mixer.mlp.fc2.bias"),
                n_patches,
            )?,
            chan_norm_w: require_tensor_1d_owned(
                weights,
                &layer_aliases(layer, "channel_norm.weight", "channel_mixer.norm.weight"),
                n_channels,
            )?,
            chan_norm_b: require_tensor_1d_owned(
                weights,
                &layer_aliases(layer, "channel_norm.bias", "channel_mixer.norm.bias"),
                n_channels,
            )?,
            chan_fc1_w: require_tensor_2d_owned(
                weights,
                &layer_aliases(
                    layer,
                    "channel_mlp.fc1.weight",
                    "channel_mixer.mlp.fc1.weight",
                ),
                n_channels,
                chan_hidden,
            )?,
            chan_fc1_b: require_tensor_1d_owned(
                weights,
                &layer_aliases(layer, "channel_mlp.fc1.bias", "channel_mixer.mlp.fc1.bias"),
                chan_hidden,
            )?,
            chan_fc2_w: require_tensor_2d_owned(
                weights,
                &layer_aliases(
                    layer,
                    "channel_mlp.fc2.weight",
                    "channel_mixer.mlp.fc2.weight",
                ),
                chan_hidden,
                n_channels,
            )?,
            chan_fc2_b: require_tensor_1d_owned(
                weights,
                &layer_aliases(layer, "channel_mlp.fc2.bias", "channel_mixer.mlp.fc2.bias"),
                n_channels,
            )?,
        })
    }

    fn as_axis_weights(
        &self,
        n_patches: usize,
        n_channels: usize,
        time_hidden: usize,
        chan_hidden: usize,
    ) -> (AxisMlpWeights<'_>, AxisMlpWeights<'_>) {
        (
            AxisMlpWeights {
                norm_weight: &self.time_norm_w,
                norm_bias: &self.time_norm_b,
                fc1_weight: &self.time_fc1_w,
                fc1_bias: &self.time_fc1_b,
                fc2_weight: &self.time_fc2_w,
                fc2_bias: &self.time_fc2_b,
                dim: n_patches,
                hidden_dim: time_hidden,
            },
            AxisMlpWeights {
                norm_weight: &self.chan_norm_w,
                norm_bias: &self.chan_norm_b,
                fc1_weight: &self.chan_fc1_w,
                fc1_bias: &self.chan_fc1_b,
                fc2_weight: &self.chan_fc2_w,
                fc2_bias: &self.chan_fc2_b,
                dim: n_channels,
                hidden_dim: chan_hidden,
            },
        )
    }
}

impl ArchRecipe for TtmRecipe {
    fn architecture_id(&self) -> &'static str {
        "ttm"
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let _ = self.validate_geometry()?;
        validate_quant_and_arch(self, metadata, weights)?;
        if weights.tensor_count() > 0 {
            self.validate_weights(weights)?;
        }
        Ok(())
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let n_patches = self.validate_geometry()?;
        let features = ctx.feature_input().ok_or_else(|| {
            bounds_err(
                "feature_input",
                self.context_length * self.num_channels,
                0,
                "ttm requires ExecutionContext::feature_input continuous telemetry matrix"
                    .to_owned(),
            )
        })?;
        let (num_rows, num_cols) = ctx.feature_dims().ok_or_else(|| {
            bounds_err(
                "feature_input",
                1,
                0,
                "ttm requires ExecutionContext::feature_dims".to_owned(),
            )
        })?;
        validate_finite_slice("feature_input", features)?;

        let out_elems = checked_mul("ttm_forecast", self.forecast_length, self.num_channels)?;
        if ctx.logits().len() < out_elems {
            return Err(bounds_err(
                "logits",
                out_elems,
                ctx.logits().len(),
                format!(
                    "ttm forecast output buffer too small: need {out_elems} ({}x{}), got {}",
                    self.forecast_length,
                    self.num_channels,
                    ctx.logits().len()
                ),
            ));
        }

        let mut state = self.prepare_patches(features, num_rows, num_cols, n_patches, weights)?;
        let req_scratch = ttm_time_channel_mix_scratch_len(
            n_patches,
            self.num_channels,
            self.time_hidden_dim,
            self.channel_hidden_dim,
        )
        .ok_or_else(|| {
            bounds_err(
                "ttm_scratch",
                0,
                0,
                "ttm scratch dimension calculation overflowed usize".to_owned(),
            )
        })?;
        let mut scratch = vec![0.0_f32; req_scratch];
        let mut next_state = vec![0.0_f32; state.len()];

        for layer in 0..self.num_mixer_layers {
            let tensors = TtmLayerTensors::load(
                weights,
                layer,
                n_patches,
                self.num_channels,
                self.time_hidden_dim,
                self.channel_hidden_dim,
            )?;
            let (time_w, chan_w) = tensors.as_axis_weights(
                n_patches,
                self.num_channels,
                self.time_hidden_dim,
                self.channel_hidden_dim,
            );
            ttm_time_channel_mix_with_scratch_f32(
                &mut next_state,
                &state,
                n_patches,
                self.num_channels,
                &time_w,
                &chan_w,
                self.norm_eps,
                &mut scratch,
            )?;
            state.copy_from_slice(&next_state);
        }

        let forecast = self.project_forecast_head(weights, &state, n_patches, out_elems)?;
        validate_finite_slice("ttm_forecast", &forecast)?;
        ctx.logits_mut()[..out_elems].copy_from_slice(&forecast);
        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.forward_classify(ctx, weights)
    }

    fn decode_step(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        Err(mismatch_err(
            "ttm",
            "TinyTimeMixers is a zero-KV single-pass forecast model and does not support autoregressive decode_step",
        ))
    }
}
