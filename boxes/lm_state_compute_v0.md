# Box `lm-state-compute` v0

61 nodes; an edge A -> B means B's recorded justification or construction uses A.

## Fixed requirements

- Next-token prediction on the frozen task corpus with its fixed tokenizer (vocabulary 4096), context length 512, and batch 32
- Causal prefix access only: the prediction at t depends only on tokens <= t of the same sequence
- No state carried across sequences, batch rows, or evaluation calls
- Equal training-FLOP budget per candidate, counted as executed, with the same data order and init seed for paired comparison
- Evaluator, data splits, and engineering checks are fixed; the confirm split is sealed
- At most 60M parameters

## Editable priors

- Matched training FLOPs is the fair comparison (rather than matched parameters or wall-clock)
- Early-checkpoint dev loss is informative about longer-run dev loss (Stage A tests this)
- Results reported at 70M-8B transfer qualitatively to 5-30M models
- Average next-token loss is the primary utility; mechanism-specific probes are diagnostics

## Permitted constructions

- Any torch modules, including custom recurrences over depth or sequence within one forward pass
- Per-token adaptive computation, with an auxiliary loss returned as aux_loss
- Parameter sharing, routing, gating, extra latent positions, or new state variables within a sequence
- Recipe changes within the allowed train_config bounds

## Active commitments of the starting formulation

**`c.fixed_depth`** (assumption, established, active): Every position receives the same computation: L blocks, each applied exactly once, in a fixed order.
  - scope: starting decoder
  - supported by: (`def.state` + `def.compute`); used by: `c.single_stream`, `c.unshared`, `m.baseline`, `t.depth_bound`, `t.uniform_compute`
  - source: arXiv:1706.03762 Vaswani+ 2017, Attention Is All You Need
  - source: arXiv:2302.13971 Touvron+ 2023, LLaMA

**`c.unshared`** (assumption, established, active): Each block has its own parameters, so parameter count grows linearly with depth and no computation is reused across depth.
  - scope: starting decoder
  - supported by: (`c.fixed_depth`); used by: `m.baseline`
  - source: arXiv:1706.03762 Vaswani+ 2017, Attention Is All You Need

**`c.token_granularity`** (assumption, established, active): The unit of state and computation is the input token: no extra latent positions (memory or register tokens), and positions are never merged, split, or skipped.
  - scope: starting decoder
  - supported by: (`def.state`); used by: `m.baseline`
  - source: arXiv:1706.03762 Vaswani+ 2017, Attention Is All You Need

**`c.kv_memory`** (assumption, established, active): Sequence memory is the uncompressed set of all past keys and values at every layer, read by causal softmax attention; memory grows linearly with context and is never summarized.
  - scope: starting decoder
  - supported by: (`def.mixing` + `e.recall_gap`); used by: `c.qk_norm`, `c.rope`, `m.baseline`, `t.kv_cost`
  - note: Justified mainly by recall: attention is the strongest known mechanism for associative recall.
  - source: arXiv:1706.03762 Vaswani+ 2017, Attention Is All You Need

**`c.single_stream`** (assumption, established, active): Information passes between blocks through one additive residual stream of width d: h^{l+1} = h^l + f_l(norm(h^l)). Block outputs are summed, never gated, weighted by depth, or routed between multiple streams.
  - scope: starting decoder
  - supported by: (`c.fixed_depth`); used by: `c.prenorm`, `m.baseline`
  - source: arXiv:1512.03385 He+ 2015, Deep Residual Learning
  - source: arXiv:1706.03762 Vaswani+ 2017, Attention Is All You Need

**`c.dense_ffn`** (assumption, established, active): Channel mixing is one dense SwiGLU feed-forward layer per block, applied identically to every token.
  - scope: starting decoder
  - supported by: (`def.mixing` + `e.glu`); used by: `m.baseline`, `t.uniform_compute`
  - source: arXiv:2002.05202 Shazeer 2020, GLU Variants Improve Transformer
  - source: arXiv:2302.13971 Touvron+ 2023, LLaMA

**`c.rope`** (method, established, active): Position enters only through rotary embeddings applied to attention queries and keys; there is no absolute position embedding.
  - scope: starting decoder
  - supported by: (`c.kv_memory`); used by: `m.baseline`
  - note: RoPE is a construction on attention queries and keys, so it depends on the attention memory.
  - source: arXiv:2104.09864 Su+ 2021, RoFormer (RoPE)
  - source: arXiv:2302.13971 Touvron+ 2023, LLaMA

**`c.prenorm`** (method, established, active): Each sublayer input is normalized with RMSNorm (pre-norm); queries, keys, and block outputs are not normalized.
  - scope: starting decoder
  - supported by: (`c.single_stream` + `e.prenorm`); used by: `m.baseline`
  - source: arXiv:1910.07467 Zhang & Sennrich 2019, RMSNorm
  - source: arXiv:2002.04745 Xiong+ 2020, On Layer Normalization in the Transformer Architecture
  - source: arXiv:2302.13971 Touvron+ 2023, LLaMA

**`c.tied`** (method, established, active): The input embedding and the output projection share one matrix.
  - scope: starting decoder
  - supported by: (`e.tied_small`); used by: `m.baseline`
  - source: arXiv:1608.05859 Press & Wolf 2016, Using the Output Embedding
  - source: arXiv:1611.01462 Inan+ 2016, Tying Word Vectors and Word Classifiers

**`c.qk_norm`** (method, established, active): Queries and keys are RMS-normalized per head before rotary embedding, bounding attention logits.
  - scope: starting decoder
  - supported by: (`c.kv_memory` + `e.instability`); used by: `m.baseline`
  - source: arXiv:2010.04245 Henry+ 2020, Query-Key Normalization
  - source: arXiv:2309.14322 Wortsman+ 2023, Small-scale proxies for training instabilities

**`m.baseline`** (method, established, active): Llama-style causal decoder (Transformer++ with QK-norm): the starting formulation, assembled from the commitments above.
  - scope: d_model 256, 6 layers, 4 heads: 5.9M parameters at vocabulary 4096
  - supported by: (`c.fixed_depth` + `c.unshared` + `c.token_granularity` + `c.kv_memory` + `c.single_stream` + `c.dense_ffn` + `c.rope` + `c.prenorm` + `c.tied` + `c.qk_norm`); used by: -
  - source: arXiv:2302.13971 Touvron+ 2023, LLaMA
  - source: arXiv:2403.17844 Poli+ 2024, Mechanistic Design and Scaling of Hybrid Architectures (MAD)
  - source: cspace profile on this harness (measured, Nov 2026)

## Consequences and theoretical results

**`t.kv_cost`** (theorem, established, active): Under c.kv_memory, attention compute per token and inference memory grow linearly with the position in the context.
  - scope: exact for softmax attention
  - supported by: (`c.kv_memory`); used by: -
  - source: arXiv:1706.03762 Vaswani+ 2017, Attention Is All You Need

**`t.uniform_compute`** (theorem, established, active): Under c.fixed_depth and c.dense_ffn, every token gets identical executed FLOPs regardless of how hard it is to predict.
  - scope: exact for the starting decoder
  - supported by: (`c.fixed_depth` + `c.dense_ffn`); used by: -
  - source: arXiv:2404.02258 Raposo+ 2024, Mixture-of-Depths

**`t.depth_bound`** (theorem, conditional, active): Under c.fixed_depth, the serial computation available to a prediction within one forward pass is bounded by the L block applications; more serial steps require more depth, iteration, or extra tokens.
  - scope: conditional on a fixed, non-iterated stack
  - supported by: (`c.fixed_depth`); used by: -
  - note: Formal circuit-complexity bounds for fixed-depth transformers exist, but no citation for them was verified in this sweep.
  - source: arXiv:1807.03819 Dehghani+ 2018, Universal Transformers
  - source: arXiv:2301.13196 Giannou+ 2023, Looped Transformers as Programmable Computers

**`t.linear_attn_is_rnn`** (theorem, established): Causal attention with a kernel feature map in place of softmax is a recurrence with a fixed-size matrix state updated by outer products; it trains in parallel and generates recurrently.
  - scope: exact
  - supported by: (`def.mixing`); used by: `alt.linear_attention`, `alt.retention`, `e.gla_parity`, `e.recall_state`, `t.fast_weights`
  - source: arXiv:2006.16236 Katharopoulos+ 2020, Transformers are RNNs
  - source: arXiv:2102.11174 Schlag+ 2021, Linear Transformers Are Secretly Fast Weight Programmers

**`t.fast_weights`** (theorem, established): Linear attention is equivalent to a fast-weight programmer; purely additive outer-product updates have limited memory capacity, which motivates error-correcting (delta-rule) updates.
  - scope: theory plus synthetic retrieval
  - supported by: (`t.linear_attn_is_rnn`); used by: `alt.delta_rule`, `e.delta_rule`
  - source: arXiv:2102.11174 Schlag+ 2021, Linear Transformers Are Secretly Fast Weight Programmers

## Definitions

**`def.state`** (definition, established, active): At each layer l the model's state for a sequence is one hidden vector h_t^l in R^d per position t; the prediction at t reads only states at positions <= t.
  - scope: causal decoders in this task
  - supported by: root; used by: `c.fixed_depth`, `c.token_granularity`
  - source: arXiv:1706.03762 Vaswani+ 2017, Attention Is All You Need

**`def.mixing`** (definition, established, active): Sequence mixing moves information between positions (attention, convolution, or recurrence); channel mixing transforms each position's vector on its own (feed-forward layers).
  - scope: sequence models generally
  - supported by: root; used by: `c.dense_ffn`, `c.kv_memory`, `t.linear_attn_is_rnn`
  - source: arXiv:2403.17844 Poli+ 2024, Mechanistic Design and Scaling of Hybrid Architectures (MAD)

**`def.compute`** (definition, established, active): Compute is FLOPs as executed: per-token cost is the FLOPs spent on one position in one forward pass; training cost is total executed training FLOPs, the fixed budget each candidate receives in this study.
  - scope: this study's accounting
  - supported by: root; used by: `c.fixed_depth`
  - source: arXiv:2001.08361 Kaplan+ 2020, Scaling Laws for Neural Language Models
  - source: cspace profile on this harness (measured, Nov 2026)

**`def.recall`** (definition, established): Associative recall: predicting a token by retrieving the value bound to a key that appeared earlier in the context (multi-query associative recall, MQAR); induction heads are the attention mechanism for it.
  - scope: in-context retrieval in LMs
  - supported by: root; used by: `e.recall_gap`, `e.recall_state`
  - source: arXiv:2312.04927 Arora+ 2023, Zoology
  - source: arXiv:2209.11895 Olsson+ 2022, In-context Learning and Induction Heads

## Empirical claims

**`e.recall_gap`** (empirical_claim, supported): Fixed-state gated-convolution models trail attention by up to 2.1 perplexity on the Pile; tokens that need associative recall (6.4% of tokens) account for about 82% of the gap; a 70M attention model beats a 1.4B gated-convolution model on recall.
  - scope: 17 models, 70M-1.4B, Pile; 2023-era gated convolutions (predates Mamba-2 and DeltaNet)
  - supported by: (`def.recall`); used by: `c.kv_memory`, `e.hybrid_recovers`
  - source: arXiv:2312.04927 Arora+ 2023, Zoology

**`e.recall_state`** (empirical_claim, supported): Recall quality trades off against recurrent state size: fixed-state models (H3, Mamba, RWKV) struggle on recall-intensive tasks, and varying window size and feature dimension traces a recall-memory frontier.
  - scope: 1.3B models
  - supported by: (`def.recall` + `t.linear_attn_is_rnn`); used by: `alt.local_hybrid`
  - source: arXiv:2402.18668 Arora+ 2024, Simple linear attention LMs balance the recall-throughput tradeoff (BASED)

**`e.hybrid_recovers`** (empirical_claim, supported): A little attention recovers most of the recall gap: input-dependent sparse-attention hybrids close 97.4% of the Pile perplexity gap, and a 125M H3+attention hybrid beats a Transformer by 1.0 perplexity on OpenWebText.
  - scope: 70M-360M (Zoology), 125M (H3)
  - supported by: (`e.recall_gap`); used by: `alt.local_hybrid`
  - source: arXiv:2312.04927 Arora+ 2023, Zoology
  - source: arXiv:2212.14052 Fu+ 2022, Hungry Hungry Hippos (H3)

**`e.gla_parity`** (empirical_claim, supported): Gated linear attention nearly matches Transformer++: WikiText perplexity 28.65 vs 28.39 at 340M (15B tokens) and 17.22 vs 16.85 at 1.3B (100B tokens).
  - scope: 340M and 1.3B
  - supported by: (`t.linear_attn_is_rnn`); used by: `alt.linear_attention`
  - source: arXiv:2312.06635 Yang+ 2023, Gated Linear Attention Transformers

**`e.delta_rule`** (empirical_claim, supported): Delta-rule updates beat additive linear attention on WikiText-103 at about 40M and 90M parameters; pure linear models still underperform Transformers on in-context retrieval at 1.3B; gated delta rules outperform Mamba2 and DeltaNet.
  - scope: about 40M-1.3B
  - supported by: (`t.fast_weights`); used by: `alt.delta_rule`
  - source: arXiv:2102.11174 Schlag+ 2021, Linear Transformers Are Secretly Fast Weight Programmers
  - source: arXiv:2406.06484 Yang+ 2024, Parallelizing Linear Transformers with the Delta Rule
  - source: arXiv:2412.06464 Yang+ 2024, Gated Delta Networks

**`e.ssm_lm`** (empirical_claim, supported): Selective state-space models match same-size Transformers on language modeling at billion scale; Mamba-2 trains 2-8x faster than Mamba while remaining competitive.
  - scope: 3B (Mamba), 8B (empirical study)
  - supported by: root; used by: `alt.ssm`
  - source: arXiv:2312.00752 Gu & Dao 2023, Mamba
  - source: arXiv:2405.21060 Dao & Gu 2024, Transformers are SSMs (Mamba-2)
  - source: arXiv:2406.07887 Waleffe+ 2024, An Empirical Study of Mamba-based Language Models

**`e.looped_match`** (empirical_claim, supported): A k-layer block looped L times nearly matches a kL-layer model on synthetic reasoning and is competitive on downstream reasoning; looped transformers fit in-context learning tasks with under 10% of the parameters.
  - scope: synthetic tasks and LM reasoning; sizes not stated in abstracts
  - supported by: root; used by: `alt.recurrent_depth`
  - note: Motivated by t.depth_bound; the evidence stands on its own experiments.
  - source: arXiv:2502.17416 Saunshi+ 2025, Reasoning with Latent Thoughts
  - source: arXiv:2311.12424 Yang+ 2023, Looped Transformers are Better at Learning Learning Algorithms

**`e.recursion_flops`** (empirical_claim, supported): A shared recursive block with learned per-token recursion depth (Mixture-of-Recursions) lowers validation perplexity at equal training FLOPs with fewer parameters.
  - scope: 135M-1.7B, compute-matched
  - supported by: root; used by: `alt.mor`
  - note: Motivated by t.uniform_compute.
  - source: arXiv:2507.10524 Bae+ 2025, Mixture-of-Recursions

**`e.adaptive_depth_lm`** (empirical_claim, supported): Learned adaptive computation (ACT) gives large gains on synthetic algorithmic tasks but small gains on character-level language modeling; adaptive per-layer looping mainly helps math reasoning, with gated memory recovering commonsense performance.
  - scope: ACT: char-level LM; adaptive loops: 2026 workshop preprint
  - supported by: root; used by: `alt.pondernet`, `alt.ut`
  - note: Motivated by t.depth_bound.
  - source: arXiv:1603.08983 Graves 2016, Adaptive Computation Time
  - source: arXiv:2603.08391 (2026), Adaptive Loops and Memory in Transformers (workshop preprint)

**`e.mod`** (empirical_claim, supported): Routing a capacity-limited subset of tokens through each block (Mixture-of-Depths) matches baseline quality at equivalent training FLOPs and wall-clock, with faster steps at sampling.
  - scope: sizes not stated in abstract
  - supported by: root; used by: `alt.mod`
  - note: Motivated by t.uniform_compute.
  - source: arXiv:2404.02258 Raposo+ 2024, Mixture-of-Depths

**`e.early_exit`** (empirical_claim, supported): Per-token early exit and self-speculative decoding cut inference compute (roughly 2-3x) while keeping quality.
  - scope: inference-time savings on Llama-family models; not rewarded by this study's training-FLOP budget
  - supported by: root; used by: `alt.early_exit`
  - note: Motivated by t.uniform_compute.
  - source: arXiv:2207.07061 Schuster+ 2022, Confident Adaptive Language Modeling
  - source: arXiv:2404.16710 Elhoushi+ 2024, LayerSkip

**`e.memory_tokens`** (empirical_claim, supported): Learnable memory tokens improve translation and language modeling; memory tokens passed between segments make a recurrent model on par with Transformer-XL; a 3.2M single-block UT with ACT reaches non-trivial Sudoku-Extreme accuracy only with memory tokens.
  - scope: mixed; the Sudoku result is one task and one preprint
  - supported by: root; used by: `alt.memory_tokens`
  - source: arXiv:2006.11527 Burtsev+ 2020, Memory Transformer
  - source: arXiv:2207.06881 Bulatov+ 2022, Recurrent Memory Transformer
  - source: arXiv:2604.21999 (2026), Universal Transformers Need Memory (single-author preprint)

**`e.residual_variants`** (empirical_claim, supported): Depth-weighted averaging of earlier block outputs (DenseFormer) improves perplexity with a few thousand extra parameters; multiple residual streams with learned connections (hyper-connections) report significant gains over residual connections; constraining them to a manifold restores the identity mapping.
  - scope: author-reported; model sizes not stated in abstracts
  - supported by: root; used by: `alt.depth_mixing`, `alt.multi_stream`
  - source: arXiv:2402.02622 Pagliardini+ 2024, DenseFormer
  - source: arXiv:2409.19606 Zhu+ 2024, Hyper-Connections
  - source: arXiv:2512.24880 Xie+ 2025, mHC: Manifold-Constrained Hyper-Connections
  - source: arXiv:2603.14833 (2026), Ablate and Rescue: residual-stream hyper-connections

**`e.sharing`** (empirical_claim, supported): Immediate block-wise weight sharing improves accuracy at no parameter cost (+0.7% and +0.8% at 125M and 350M); cross-layer sharing cuts parameters substantially in BERT-scale encoders.
  - scope: 125M-350M decoders; BERT-scale encoders
  - supported by: root; used by: `alt.layer_sharing`
  - source: arXiv:2402.14905 Liu+ 2024, MobileLLM
  - source: arXiv:1909.11942 Lan+ 2019, ALBERT
  - source: arXiv:2104.06022 Takase & Kiyono 2021, Lessons on Parameter Sharing across Layers

**`e.shape`** (empirical_claim, disputed): Evidence on depth versus width conflicts: over a wide range shape matters little relative to parameter count (Kaplan); at fixed parameters deeper models generalize more compositionally (41M-374M); deep-and-thin is chosen at 125M-350M.
  - scope: 41M-1B+; nothing at 5-30M
  - supported by: root; used by: -
  - source: arXiv:2001.08361 Kaplan+ 2020, Scaling Laws for Neural Language Models
  - source: arXiv:2310.19956 Petty+ 2023, Impact of Depth on Compositional Generalization
  - source: arXiv:2402.14905 Liu+ 2024, MobileLLM

**`e.tinystories`** (empirical_claim, supported): Models under 10M parameters trained on TinyStories produce coherent English.
  - scope: TinyStories corpus
  - supported by: root; used by: -
  - source: arXiv:2305.07759 Eldan & Li 2023, TinyStories

**`e.moe_granularity`** (empirical_claim, supported): Fine-grained experts beat feed-forward-sized experts across compute budgets, and the dense-versus-MoE efficiency gap widens with scale.
  - scope: about 129M-3.7B on C4; no evidence at 5-30M
  - supported by: root; used by: `alt.moe`
  - source: arXiv:2402.07871 Krajewski+ 2024, Scaling Laws for Fine-Grained MoE
  - source: arXiv:2101.03961 Fedus+ 2021, Switch Transformers

**`e.glu`** (empirical_claim, supported): GLU-family feed-forward layers (SwiGLU, GEGLU) improve over ReLU and GELU feed-forward layers.
  - scope: T5-style models
  - supported by: root; used by: `c.dense_ffn`
  - source: arXiv:2002.05202 Shazeer 2020, GLU Variants Improve Transformer

**`e.prenorm`** (empirical_claim, supported): Pre-norm removes the learning-rate warmup that post-norm needs; RMSNorm matches LayerNorm and is cheaper.
  - scope: translation and pretraining experiments
  - supported by: root; used by: `c.prenorm`
  - source: arXiv:2002.04745 Xiong+ 2020, On Layer Normalization in the Transformer Architecture
  - source: arXiv:1910.07467 Zhang & Sennrich 2019, RMSNorm

**`e.instability`** (empirical_claim, supported): Large-scale instabilities (attention-logit growth, output-logit divergence) reproduce in small models at high learning rates, and QK-normalization and z-loss mitigate them.
  - scope: small models
  - supported by: root; used by: `alt.z_loss`, `c.qk_norm`
  - source: arXiv:2309.14322 Wortsman+ 2023, Small-scale proxies for training instabilities
  - source: arXiv:2010.04245 Henry+ 2020, Query-Key Normalization

**`e.position`** (empirical_claim, supported): Causal LMs without positional encoding remain competitive and learn position implicitly; no positional encoding length-generalizes better than RoPE or ALiBi on downstream reasoning; ALiBi extrapolates beyond its training length.
  - scope: various; ALiBi at 1.3B
  - supported by: root; used by: `alt.position`
  - source: arXiv:2203.16634 Haviv+ 2022, Transformer LMs without Positional Encodings
  - source: arXiv:2305.19466 Kazemnejad+ 2023, Positional Encoding and Length Generalization
  - source: arXiv:2108.12409 Press+ 2021, ALiBi

**`e.tied_small`** (empirical_claim, supported): At small scale embeddings are a large share of parameters (18% of the 5.9M baseline at vocabulary 4096), and tying is standard in small-model recipes.
  - scope: this harness; 125M-350M recipes
  - supported by: root; used by: `c.tied`
  - source: cspace profile on this harness (measured, Nov 2026)
  - source: arXiv:2402.14905 Liu+ 2024, MobileLLM
  - source: arXiv:1608.05859 Press & Wolf 2016, Using the Output Embedding

**`e.proxy_limits`** (empirical_claim, disputed): Average perplexity can hide recall deficits, and simple synthetic recall tests can miss language-model behavior; MAD's synthetic tasks are reported to predict compute-optimal perplexity.
  - scope: 70M-7B (MAD)
  - supported by: root; used by: -
  - source: arXiv:2312.04927 Arora+ 2023, Zoology
  - source: arXiv:2403.17844 Poli+ 2024, Mechanistic Design and Scaling of Hybrid Architectures (MAD)

## Known alternatives (background)

**`alt.ut`** (method, established): Universal Transformer: one shared block applied recurrently over depth with per-position dynamic halting.
  - scope: algorithmic tasks, LAMBADA, WMT
  - supported by: (`e.adaptive_depth_lm`); used by: -
  - note: Motivated by t.depth_bound.
  - source: arXiv:1807.03819 Dehghani+ 2018, Universal Transformers
  - source: arXiv:1603.08983 Graves 2016, Adaptive Computation Time

**`alt.pondernet`** (method, established): PonderNet: probabilistic halting that learns the number of steps end to end.
  - scope: synthetic tasks, QA
  - supported by: (`e.adaptive_depth_lm`); used by: -
  - source: arXiv:2107.05407 Banino+ 2021, PonderNet

**`alt.recurrent_depth`** (method, established): Recurrent-depth LM: prelude, a core block iterated a variable number of times, and coda; test-time compute scales in latent space.
  - scope: 3.5B
  - supported by: (`e.looped_match`); used by: -
  - source: arXiv:2502.05171 Geiping+ 2025, Recurrent Depth Approach
  - source: arXiv:2502.17416 Saunshi+ 2025, Reasoning with Latent Thoughts
  - source: arXiv:2301.13196 Giannou+ 2023, Looped Transformers as Programmable Computers

**`alt.mor`** (method, established): Mixture-of-Recursions: shared recursive block with per-token routing to different recursion depths.
  - scope: 135M-1.7B
  - supported by: (`e.recursion_flops`); used by: -
  - source: arXiv:2507.10524 Bae+ 2025, Mixture-of-Recursions
  - source: arXiv:2410.20672 Bae+ 2024, Relaxed Recursive Transformers

**`alt.mod`** (method, established): Mixture-of-Depths: a router sends only top-k tokens through each block; others skip it on the residual stream.
  - scope: see e.mod
  - supported by: (`e.mod`); used by: -
  - source: arXiv:2404.02258 Raposo+ 2024, Mixture-of-Depths

**`alt.early_exit`** (method, established): Early exit: tokens stop at an intermediate layer once confident.
  - scope: inference
  - supported by: (`e.early_exit`); used by: -
  - source: arXiv:2207.07061 Schuster+ 2022, Confident Adaptive Language Modeling
  - source: arXiv:2404.16710 Elhoushi+ 2024, LayerSkip

**`alt.linear_attention`** (method, established): Linear or gated linear attention: a fixed-size matrix state replaces the KV history; log-linear variants grow the state logarithmically.
  - scope: see e.gla_parity
  - supported by: (`t.linear_attn_is_rnn` + `e.gla_parity`); used by: -
  - source: arXiv:2006.16236 Katharopoulos+ 2020, Transformers are RNNs
  - source: arXiv:2312.06635 Yang+ 2023, Gated Linear Attention Transformers
  - source: arXiv:2506.04761 Log-Linear Attention (2025)

**`alt.delta_rule`** (method, established): Delta-rule state updates (DeltaNet, Gated DeltaNet, RWKV-7): the state is corrected toward new key-value pairs instead of summed.
  - scope: see e.delta_rule
  - supported by: (`t.fast_weights` + `e.delta_rule`); used by: -
  - source: arXiv:2406.06484 Yang+ 2024, Parallelizing Linear Transformers with the Delta Rule
  - source: arXiv:2412.06464 Yang+ 2024, Gated Delta Networks
  - source: arXiv:2503.14456 Peng+ 2025, RWKV-7 Goose

**`alt.ssm`** (method, established): State-space sequence mixing (S4, Mamba, Mamba-2): input-dependent linear recurrence with a fixed-size state.
  - scope: see e.ssm_lm
  - supported by: (`e.ssm_lm`); used by: -
  - source: arXiv:2111.00396 Gu+ 2021, Structured State Spaces (S4)
  - source: arXiv:2312.00752 Gu & Dao 2023, Mamba
  - source: arXiv:2405.21060 Dao & Gu 2024, Transformers are SSMs (Mamba-2)

**`alt.retention`** (method, established): Retention and RWKV-style recurrences with parallel training and recurrent inference.
  - scope: language modeling up to 14B
  - supported by: (`t.linear_attn_is_rnn`); used by: -
  - source: arXiv:2307.08621 Sun+ 2023, Retentive Network
  - source: arXiv:2305.13048 Peng+ 2023, RWKV

**`alt.local_hybrid`** (method, established): Local (sliding-window) attention combined with a recurrent or linear-attention layer (Griffin, Samba, BASED, Jamba).
  - scope: 125M-14B
  - supported by: (`e.hybrid_recovers` + `e.recall_state`); used by: -
  - source: arXiv:2004.05150 Beltagy+ 2020, Longformer
  - source: arXiv:2402.19427 De+ 2024, Griffin
  - source: arXiv:2406.07522 Ren+ 2024, Samba
  - source: arXiv:2402.18668 Arora+ 2024, Simple linear attention LMs balance the recall-throughput tradeoff (BASED)
  - source: arXiv:2403.19887 Lieber+ 2024, Jamba

**`alt.memory_tokens`** (method, established): Memory or register tokens: extra latent positions that read and write global state.
  - scope: see e.memory_tokens
  - supported by: (`e.memory_tokens`); used by: -
  - source: arXiv:2006.11527 Burtsev+ 2020, Memory Transformer
  - source: arXiv:2207.06881 Bulatov+ 2022, Recurrent Memory Transformer
  - source: arXiv:2309.16588 Darcet+ 2023, Vision Transformers Need Registers

**`alt.multi_stream`** (method, established): Hyper-connections: several residual streams with learned depth and width connections.
  - scope: see e.residual_variants
  - supported by: (`e.residual_variants`); used by: -
  - source: arXiv:2409.19606 Zhu+ 2024, Hyper-Connections
  - source: arXiv:2512.24880 Xie+ 2025, mHC: Manifold-Constrained Hyper-Connections

**`alt.depth_mixing`** (method, established): Learned access to earlier layers: depth-weighted averaging (DenseFormer) or gated highway paths.
  - scope: see e.residual_variants
  - supported by: (`e.residual_variants`); used by: -
  - source: arXiv:2402.02622 Pagliardini+ 2024, DenseFormer
  - source: arXiv:1505.00387 Srivastava+ 2015, Highway Networks

**`alt.layer_sharing`** (method, established): Cross-layer or block-wise weight sharing.
  - scope: see e.sharing
  - supported by: (`e.sharing`); used by: -
  - source: arXiv:1909.11942 Lan+ 2019, ALBERT
  - source: arXiv:2402.14905 Liu+ 2024, MobileLLM
  - source: arXiv:2104.06022 Takase & Kiyono 2021, Lessons on Parameter Sharing across Layers

**`alt.moe`** (method, established): Sparse mixture-of-experts feed-forward layers with learned routing.
  - scope: see e.moe_granularity
  - supported by: (`e.moe_granularity`); used by: -
  - source: arXiv:1701.06538 Shazeer+ 2017, Sparsely-Gated Mixture-of-Experts
  - source: arXiv:2101.03961 Fedus+ 2021, Switch Transformers
  - source: arXiv:2409.02060 Muennighoff+ 2024, OLMoE

**`alt.position`** (method, established): ALiBi distance biases or no positional encoding (NoPE).
  - scope: see e.position
  - supported by: (`e.position`); used by: -
  - source: arXiv:2108.12409 Press+ 2021, ALiBi
  - source: arXiv:2203.16634 Haviv+ 2022, Transformer LMs without Positional Encodings

**`alt.z_loss`** (method, established): Output z-loss: an auxiliary penalty on the log-partition of the output logits for logit stability.
  - scope: see e.instability
  - supported by: (`e.instability`); used by: -
  - source: arXiv:2309.14322 Wortsman+ 2023, Small-scale proxies for training instabilities

## Sources

- arXiv:1505.00387 Srivastava+ 2015, Highway Networks
- arXiv:1512.03385 He+ 2015, Deep Residual Learning
- arXiv:1603.08983 Graves 2016, Adaptive Computation Time
- arXiv:1608.05859 Press & Wolf 2016, Using the Output Embedding
- arXiv:1611.01462 Inan+ 2016, Tying Word Vectors and Word Classifiers
- arXiv:1701.06538 Shazeer+ 2017, Sparsely-Gated Mixture-of-Experts
- arXiv:1706.03762 Vaswani+ 2017, Attention Is All You Need
- arXiv:1807.03819 Dehghani+ 2018, Universal Transformers
- arXiv:1909.11942 Lan+ 2019, ALBERT
- arXiv:1910.07467 Zhang & Sennrich 2019, RMSNorm
- arXiv:2001.08361 Kaplan+ 2020, Scaling Laws for Neural Language Models
- arXiv:2002.04745 Xiong+ 2020, On Layer Normalization in the Transformer Architecture
- arXiv:2002.05202 Shazeer 2020, GLU Variants Improve Transformer
- arXiv:2004.05150 Beltagy+ 2020, Longformer
- arXiv:2006.11527 Burtsev+ 2020, Memory Transformer
- arXiv:2006.16236 Katharopoulos+ 2020, Transformers are RNNs
- arXiv:2010.04245 Henry+ 2020, Query-Key Normalization
- arXiv:2101.03961 Fedus+ 2021, Switch Transformers
- arXiv:2102.11174 Schlag+ 2021, Linear Transformers Are Secretly Fast Weight Programmers
- arXiv:2104.06022 Takase & Kiyono 2021, Lessons on Parameter Sharing across Layers
- arXiv:2104.09864 Su+ 2021, RoFormer (RoPE)
- arXiv:2107.05407 Banino+ 2021, PonderNet
- arXiv:2108.12409 Press+ 2021, ALiBi
- arXiv:2111.00396 Gu+ 2021, Structured State Spaces (S4)
- arXiv:2203.16634 Haviv+ 2022, Transformer LMs without Positional Encodings
- arXiv:2207.06881 Bulatov+ 2022, Recurrent Memory Transformer
- arXiv:2207.07061 Schuster+ 2022, Confident Adaptive Language Modeling
- arXiv:2209.11895 Olsson+ 2022, In-context Learning and Induction Heads
- arXiv:2212.14052 Fu+ 2022, Hungry Hungry Hippos (H3)
- arXiv:2301.13196 Giannou+ 2023, Looped Transformers as Programmable Computers
- arXiv:2302.13971 Touvron+ 2023, LLaMA
- arXiv:2305.07759 Eldan & Li 2023, TinyStories
- arXiv:2305.13048 Peng+ 2023, RWKV
- arXiv:2305.19466 Kazemnejad+ 2023, Positional Encoding and Length Generalization
- arXiv:2307.08621 Sun+ 2023, Retentive Network
- arXiv:2309.14322 Wortsman+ 2023, Small-scale proxies for training instabilities
- arXiv:2309.16588 Darcet+ 2023, Vision Transformers Need Registers
- arXiv:2310.19956 Petty+ 2023, Impact of Depth on Compositional Generalization
- arXiv:2311.12424 Yang+ 2023, Looped Transformers are Better at Learning Learning Algorithms
- arXiv:2312.00752 Gu & Dao 2023, Mamba
- arXiv:2312.04927 Arora+ 2023, Zoology
- arXiv:2312.06635 Yang+ 2023, Gated Linear Attention Transformers
- arXiv:2402.02622 Pagliardini+ 2024, DenseFormer
- arXiv:2402.07871 Krajewski+ 2024, Scaling Laws for Fine-Grained MoE
- arXiv:2402.14905 Liu+ 2024, MobileLLM
- arXiv:2402.18668 Arora+ 2024, Simple linear attention LMs balance the recall-throughput tradeoff (BASED)
- arXiv:2402.19427 De+ 2024, Griffin
- arXiv:2403.17844 Poli+ 2024, Mechanistic Design and Scaling of Hybrid Architectures (MAD)
- arXiv:2403.19887 Lieber+ 2024, Jamba
- arXiv:2404.02258 Raposo+ 2024, Mixture-of-Depths
- arXiv:2404.16710 Elhoushi+ 2024, LayerSkip
- arXiv:2405.21060 Dao & Gu 2024, Transformers are SSMs (Mamba-2)
- arXiv:2406.06484 Yang+ 2024, Parallelizing Linear Transformers with the Delta Rule
- arXiv:2406.07522 Ren+ 2024, Samba
- arXiv:2406.07887 Waleffe+ 2024, An Empirical Study of Mamba-based Language Models
- arXiv:2409.02060 Muennighoff+ 2024, OLMoE
- arXiv:2409.19606 Zhu+ 2024, Hyper-Connections
- arXiv:2410.20672 Bae+ 2024, Relaxed Recursive Transformers
- arXiv:2412.06464 Yang+ 2024, Gated Delta Networks
- arXiv:2502.05171 Geiping+ 2025, Recurrent Depth Approach
- arXiv:2502.17416 Saunshi+ 2025, Reasoning with Latent Thoughts
- arXiv:2503.14456 Peng+ 2025, RWKV-7 Goose
- arXiv:2506.04761 Log-Linear Attention (2025)
- arXiv:2507.10524 Bae+ 2025, Mixture-of-Recursions
- arXiv:2512.24880 Xie+ 2025, mHC: Manifold-Constrained Hyper-Connections
- arXiv:2603.08391 (2026), Adaptive Loops and Memory in Transformers (workshop preprint)
- arXiv:2603.14833 (2026), Ablate and Rescue: residual-stream hyper-connections
- arXiv:2604.21999 (2026), Universal Transformers Need Memory (single-author preprint)
- cspace profile on this harness (measured, Nov 2026)
