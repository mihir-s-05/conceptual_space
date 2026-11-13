# Seed: Llama-style decoder

Starting formulation for the architecture pilot, following the Transformer++
recipe used as a baseline in MAD (Poli et al., 2024): pre-norm RMSNorm, rotary
position embeddings, SwiGLU MLP, causal softmax attention, tied input/output
embeddings, no biases. Queries and keys are RMS-normalized per head before RoPE
(QK-norm, as in Qwen3, Gemma 3 and OLMo 2) for attention-logit stability.

Default size (`CONFIG`): d_model=256, 6 layers, 4 heads. With a 4,096-token
vocabulary this is about 1.05M embedding and 4.8M non-embedding parameters.
Change `CONFIG` to scale; the final size is chosen after profiling.
