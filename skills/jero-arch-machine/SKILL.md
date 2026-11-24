---
name: jero-arch-machine
description: Non-tokenized machine-model architecture recipes (`TTM`, `FT-Transformer`, `TabPFN-v2`, `DecisionTransformer`) and `MmprojLatentBridge` for the SpacemiT K3 platform.
---

# `jero-arch-machine` Architecture Crate

Provides non-tokenized continuous-feature execution recipes (`TtmRecipe`, `FtTransformerRecipe`, `TabPfnV2Recipe`, `DecisionTransformerRecipe`) and the multimodal soft-prompt projector (`MmprojLatentBridge`, `project_machine_latent_to_soft_prompt`, `forward_with_soft_prompt`) with native `GgufFile` and `SafeTensorsFile` (`ModelWeights`) support.

## Exported Recipes & Projector API

- **`TtmRecipe` (`"ttm"`, zero-KV)**: Patch-time and channel-mixing multivariate time-series forecaster hydrated via `from_metadata`, `from_gguf`, or `from_safetensors` (supporting GGUF `ttm.*` / `blk.{l}.*` and HuggingFace `backbone.*` / `head.projection.*` tensor names).
- **`FtTransformerRecipe` (`"ft-transformer"`, zero-KV)**: Continuous + categorical tabular feature tokenizer and bidirectional Transformer encoder hydrated via `from_metadata`, `from_gguf`, or `from_safetensors` (supporting GGUF `ft.*` and `rtdl` `feature_tokenizer.*` / `transformer.*` aliases).
- **`TabPfnV2Recipe` (`"tabpfn-v2"`, zero-KV)**: Dual-axis (`Feature` + context-restricted `Sample`) prior-data in-context tabular classifier hydrated via `from_metadata`, `from_gguf`, or `from_safetensors` (supporting `tabpfn.*` and `transformer_encoder.*` / `decoder_*` aliases).
- **`DecisionTransformerRecipe` (`"decision-transformer"`, KV-cached)**: Causal `(return_to_go, state, action)` offline-RL trajectory transformer with GQA, NeoX `RoPE`, `RMSNorm`, `SwiGLU`, and Layer-0 soft-prompt prefix support (`from_metadata`, `from_gguf`, `from_safetensors`).
- **`MmprojLatentBridge`**: Projects continuous machine-model latents (`[machine_latent_dim]`) via 2-layer `GELU` MLP (`mmproj.fc1/fc2` or `mmproj.0/2` / `multi_modal_projector.*`) or single linear projection + optional `LayerNorm` into LLM Layer-0 soft-prompt embeddings (`forward_with_soft_prompt`), validating `(architecture_id, recipe_version, embedding_length)` tuple compatibility upfront.

## Crate Contract & Invariants

- **License**: `LGPL-3.0-only WITH LGPL-3.0-linking-exception`
- **Upstream Dependencies**: `jero-core` (`ModelWeights`, `ExecutionContext`, `ArchRecipe`, `GgufFile`, `SafeTensorsFile`, `K3Error`) and optional `jero-kernels-rvv` (enabled by default via `with-kernels`, with portable scalar `F32`/`F16`/`BF16` fallbacks under `--no-default-features`). Never depends on `jero-engine`.
- **ADR-045 Fail-Closed Enforcement**: All active forward paths reject missing tensors, mismatched GGUF/SafeTensors matrix shapes, unsupported quantizations, undersized KV/logit buffers, and non-finite (`NaN`/`±Inf`) inputs or weights (`F32`, `F16`, `BF16`) with structured `K3Error` diagnostics — zero synthetic weight or identity fallbacks.
- **Safety & Lints**: `#![deny(unsafe_op_in_unsafe_fn)]`, `clippy::pedantic`, and `clippy::undocumented_unsafe_blocks`.

## Local Verification Gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo check --target riscv64gc-unknown-linux-gnu --all-targets
cargo test --lib --tests
cargo test --no-default-features --lib --tests
MIRIFLAGS="-Zmiri-disable-isolation" cargo +nightly miri test --lib
cargo agents plugin validate Symposium.toml --no-check-crates
```
