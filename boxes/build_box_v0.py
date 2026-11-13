"""Generate boxes/lm_state_compute_v0.json from the verified citation sweep (boxes/research/).

This produced the first draft. Edit the JSON directly from now on, or edit here and regenerate."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cspace.box import validate_space  # noqa: E402

S = {  # every arXiv ID below was checked against its abstract page (title + first author)
    "vaswani": "arXiv:1706.03762 Vaswani+ 2017, Attention Is All You Need",
    "llama": "arXiv:2302.13971 Touvron+ 2023, LLaMA",
    "mad": "arXiv:2403.17844 Poli+ 2024, Mechanistic Design and Scaling of Hybrid Architectures (MAD)",
    "kaplan": "arXiv:2001.08361 Kaplan+ 2020, Scaling Laws for Neural Language Models",
    "zoology": "arXiv:2312.04927 Arora+ 2023, Zoology",
    "olsson": "arXiv:2209.11895 Olsson+ 2022, In-context Learning and Induction Heads",
    "resnet": "arXiv:1512.03385 He+ 2015, Deep Residual Learning",
    "highway": "arXiv:1505.00387 Srivastava+ 2015, Highway Networks",
    "glu": "arXiv:2002.05202 Shazeer 2020, GLU Variants Improve Transformer",
    "roformer": "arXiv:2104.09864 Su+ 2021, RoFormer (RoPE)",
    "rmsnorm": "arXiv:1910.07467 Zhang & Sennrich 2019, RMSNorm",
    "xiong": "arXiv:2002.04745 Xiong+ 2020, On Layer Normalization in the Transformer Architecture",
    "presswolf": "arXiv:1608.05859 Press & Wolf 2016, Using the Output Embedding",
    "inan": "arXiv:1611.01462 Inan+ 2016, Tying Word Vectors and Word Classifiers",
    "katharopoulos": "arXiv:2006.16236 Katharopoulos+ 2020, Transformers are RNNs",
    "schlag": "arXiv:2102.11174 Schlag+ 2021, Linear Transformers Are Secretly Fast Weight Programmers",
    "gla": "arXiv:2312.06635 Yang+ 2023, Gated Linear Attention Transformers",
    "loglinear": "arXiv:2506.04761 Log-Linear Attention (2025)",
    "deltanet": "arXiv:2406.06484 Yang+ 2024, Parallelizing Linear Transformers with the Delta Rule",
    "gdeltanet": "arXiv:2412.06464 Yang+ 2024, Gated Delta Networks",
    "s4": "arXiv:2111.00396 Gu+ 2021, Structured State Spaces (S4)",
    "mamba": "arXiv:2312.00752 Gu & Dao 2023, Mamba",
    "mamba2": "arXiv:2405.21060 Dao & Gu 2024, Transformers are SSMs (Mamba-2)",
    "waleffe": "arXiv:2406.07887 Waleffe+ 2024, An Empirical Study of Mamba-based Language Models",
    "retnet": "arXiv:2307.08621 Sun+ 2023, Retentive Network",
    "rwkv": "arXiv:2305.13048 Peng+ 2023, RWKV",
    "rwkv7": "arXiv:2503.14456 Peng+ 2025, RWKV-7 Goose",
    "longformer": "arXiv:2004.05150 Beltagy+ 2020, Longformer",
    "griffin": "arXiv:2402.19427 De+ 2024, Griffin",
    "samba": "arXiv:2406.07522 Ren+ 2024, Samba",
    "jamba": "arXiv:2403.19887 Lieber+ 2024, Jamba",
    "based": "arXiv:2402.18668 Arora+ 2024, Simple linear attention LMs balance the recall-throughput tradeoff (BASED)",
    "h3": "arXiv:2212.14052 Fu+ 2022, Hungry Hungry Hippos (H3)",
    "act": "arXiv:1603.08983 Graves 2016, Adaptive Computation Time",
    "ut": "arXiv:1807.03819 Dehghani+ 2018, Universal Transformers",
    "pondernet": "arXiv:2107.05407 Banino+ 2021, PonderNet",
    "geiping": "arXiv:2502.05171 Geiping+ 2025, Recurrent Depth Approach",
    "giannou": "arXiv:2301.13196 Giannou+ 2023, Looped Transformers as Programmable Computers",
    "yanglooped": "arXiv:2311.12424 Yang+ 2023, Looped Transformers are Better at Learning Learning Algorithms",
    "saunshi": "arXiv:2502.17416 Saunshi+ 2025, Reasoning with Latent Thoughts",
    "mor": "arXiv:2507.10524 Bae+ 2025, Mixture-of-Recursions",
    "relaxedrec": "arXiv:2410.20672 Bae+ 2024, Relaxed Recursive Transformers",
    "adaptiveloops": "arXiv:2603.08391 (2026), Adaptive Loops and Memory in Transformers (workshop preprint)",
    "mod": "arXiv:2404.02258 Raposo+ 2024, Mixture-of-Depths",
    "calm": "arXiv:2207.07061 Schuster+ 2022, Confident Adaptive Language Modeling",
    "layerskip": "arXiv:2404.16710 Elhoushi+ 2024, LayerSkip",
    "burtsev": "arXiv:2006.11527 Burtsev+ 2020, Memory Transformer",
    "rmt": "arXiv:2207.06881 Bulatov+ 2022, Recurrent Memory Transformer",
    "registers": "arXiv:2309.16588 Darcet+ 2023, Vision Transformers Need Registers",
    "utmemory": "arXiv:2604.21999 (2026), Universal Transformers Need Memory (single-author preprint)",
    "denseformer": "arXiv:2402.02622 Pagliardini+ 2024, DenseFormer",
    "hc": "arXiv:2409.19606 Zhu+ 2024, Hyper-Connections",
    "mhc": "arXiv:2512.24880 Xie+ 2025, mHC: Manifold-Constrained Hyper-Connections",
    "ablaterescue": "arXiv:2603.14833 (2026), Ablate and Rescue: residual-stream hyper-connections",
    "albert": "arXiv:1909.11942 Lan+ 2019, ALBERT",
    "mobilellm": "arXiv:2402.14905 Liu+ 2024, MobileLLM",
    "takase": "arXiv:2104.06022 Takase & Kiyono 2021, Lessons on Parameter Sharing across Layers",
    "petty": "arXiv:2310.19956 Petty+ 2023, Impact of Depth on Compositional Generalization",
    "tinystories": "arXiv:2305.07759 Eldan & Li 2023, TinyStories",
    "moe": "arXiv:1701.06538 Shazeer+ 2017, Sparsely-Gated Mixture-of-Experts",
    "switch": "arXiv:2101.03961 Fedus+ 2021, Switch Transformers",
    "finemoe": "arXiv:2402.07871 Krajewski+ 2024, Scaling Laws for Fine-Grained MoE",
    "olmoe": "arXiv:2409.02060 Muennighoff+ 2024, OLMoE",
    "alibi": "arXiv:2108.12409 Press+ 2021, ALiBi",
    "haviv": "arXiv:2203.16634 Haviv+ 2022, Transformer LMs without Positional Encodings",
    "kazemnejad": "arXiv:2305.19466 Kazemnejad+ 2023, Positional Encoding and Length Generalization",
    "qknorm": "arXiv:2010.04245 Henry+ 2020, Query-Key Normalization",
    "wortsman": "arXiv:2309.14322 Wortsman+ 2023, Small-scale proxies for training instabilities",
    "harness": "cspace profile on this harness (measured, Nov 2026)",
}


def n(nid, ntype, statement, scope, sources, status="established", active=False, supports=(), notes=None):
    node = dict(id=nid, type=ntype, statement=statement, scope=scope, sources=[S[s] for s in sources],
                status=status, active=active, supports=[list(g) for g in supports])
    if notes:
        node["notes"] = notes
    return node


nodes = [
    # ---------------- definitions ----------------
    n("def.state", "definition",
      "At each layer l the model's state for a sequence is one hidden vector h_t^l in R^d per position t; the "
      "prediction at t reads only states at positions <= t.", "causal decoders in this task", ["vaswani"], active=True),
    n("def.mixing", "definition",
      "Sequence mixing moves information between positions (attention, convolution, or recurrence); channel mixing "
      "transforms each position's vector on its own (feed-forward layers).", "sequence models generally", ["mad"],
      active=True),
    n("def.compute", "definition",
      "Compute is FLOPs as executed: per-token cost is the FLOPs spent on one position in one forward pass; training "
      "cost is total executed training FLOPs, the fixed budget each candidate receives in this study.",
      "this study's accounting", ["kaplan", "harness"], active=True),
    n("def.recall", "definition",
      "Associative recall: predicting a token by retrieving the value bound to a key that appeared earlier in the "
      "context (multi-query associative recall, MQAR); induction heads are the attention mechanism for it.",
      "in-context retrieval in LMs", ["zoology", "olsson"]),

    # ---------------- active commitments of the starting formulation ----------------
    n("c.fixed_depth", "assumption",
      "Every position receives the same computation: L blocks, each applied exactly once, in a fixed order.",
      "starting decoder", ["vaswani", "llama"], active=True, supports=[["def.state", "def.compute"]]),
    n("c.unshared", "assumption",
      "Each block has its own parameters, so parameter count grows linearly with depth and no computation is reused "
      "across depth.", "starting decoder", ["vaswani"], active=True, supports=[["c.fixed_depth"]]),
    n("c.token_granularity", "assumption",
      "The unit of state and computation is the input token: no extra latent positions (memory or register tokens), "
      "and positions are never merged, split, or skipped.", "starting decoder", ["vaswani"], active=True,
      supports=[["def.state"]]),
    n("c.kv_memory", "assumption",
      "Sequence memory is the uncompressed set of all past keys and values at every layer, read by causal softmax "
      "attention; memory grows linearly with context and is never summarized.", "starting decoder",
      ["vaswani"], active=True, supports=[["def.mixing", "e.recall_gap"]],
      notes="Justified mainly by recall: attention is the strongest known mechanism for associative recall."),
    n("c.single_stream", "assumption",
      "Information passes between blocks through one additive residual stream of width d: "
      "h^{l+1} = h^l + f_l(norm(h^l)). Block outputs are summed, never gated, weighted by depth, or routed between "
      "multiple streams.", "starting decoder", ["resnet", "vaswani"], active=True, supports=[["c.fixed_depth"]]),
    n("c.dense_ffn", "assumption",
      "Channel mixing is one dense SwiGLU feed-forward layer per block, applied identically to every token.",
      "starting decoder", ["glu", "llama"], active=True, supports=[["def.mixing", "e.glu"]]),
    n("c.rope", "method",
      "Position enters only through rotary embeddings applied to attention queries and keys; there is no absolute "
      "position embedding.", "starting decoder", ["roformer", "llama"], active=True, supports=[["c.kv_memory"]],
      notes="RoPE is a construction on attention queries and keys, so it depends on the attention memory."),
    n("c.prenorm", "method",
      "Each sublayer input is normalized with RMSNorm (pre-norm); queries, keys, and block outputs are not "
      "normalized.", "starting decoder", ["rmsnorm", "xiong", "llama"], active=True,
      supports=[["c.single_stream", "e.prenorm"]]),
    n("c.tied", "method",
      "The input embedding and the output projection share one matrix.", "starting decoder",
      ["presswolf", "inan"], active=True, supports=[["e.tied_small"]]),
    n("m.baseline", "method",
      "Llama-style causal decoder (Transformer++): the starting formulation, assembled from the commitments above.",
      "d_model 256, 6 layers, 4 heads: 5.9M parameters at vocabulary 4096", ["llama", "mad", "harness"], active=True,
      supports=[["c.fixed_depth", "c.unshared", "c.token_granularity", "c.kv_memory", "c.single_stream",
                 "c.dense_ffn", "c.rope", "c.prenorm", "c.tied"]]),

    # ---------------- consequences and theoretical results ----------------
    n("t.kv_cost", "theorem",
      "Under c.kv_memory, attention compute per token and inference memory grow linearly with the position in the "
      "context.", "exact for softmax attention", ["vaswani"], active=True, supports=[["c.kv_memory"]]),
    n("t.uniform_compute", "theorem",
      "Under c.fixed_depth and c.dense_ffn, every token gets identical executed FLOPs regardless of how hard it is "
      "to predict.", "exact for the starting decoder", ["mod"], active=True,
      supports=[["c.fixed_depth", "c.dense_ffn"]]),
    n("t.depth_bound", "theorem",
      "Under c.fixed_depth, the serial computation available to a prediction within one forward pass is bounded by "
      "the L block applications; more serial steps require more depth, iteration, or extra tokens.",
      "conditional on a fixed, non-iterated stack", ["ut", "giannou"], status="conditional", active=True,
      supports=[["c.fixed_depth"]],
      notes="Formal circuit-complexity bounds for fixed-depth transformers exist, but no citation for them was "
            "verified in this sweep."),
    n("t.linear_attn_is_rnn", "theorem",
      "Causal attention with a kernel feature map in place of softmax is a recurrence with a fixed-size matrix state "
      "updated by outer products; it trains in parallel and generates recurrently.", "exact",
      ["katharopoulos", "schlag"], supports=[["def.mixing"]]),
    n("t.fast_weights", "theorem",
      "Linear attention is equivalent to a fast-weight programmer; purely additive outer-product updates have "
      "limited memory capacity, which motivates error-correcting (delta-rule) updates.",
      "theory plus synthetic retrieval", ["schlag"], supports=[["t.linear_attn_is_rnn"]]),

    # ---------------- empirical claims (background knowledge, with scale) ----------------
    n("e.recall_gap", "empirical_claim",
      "Fixed-state gated-convolution models trail attention by up to 2.1 perplexity on the Pile; tokens that need "
      "associative recall (6.4% of tokens) account for about 82% of the gap; a 70M attention model beats a 1.4B "
      "gated-convolution model on recall.",
      "17 models, 70M-1.4B, Pile; 2023-era gated convolutions (predates Mamba-2 and DeltaNet)", ["zoology"],
      status="supported", supports=[["def.recall"]]),
    n("e.recall_state", "empirical_claim",
      "Recall quality trades off against recurrent state size: fixed-state models (H3, Mamba, RWKV) struggle on "
      "recall-intensive tasks, and varying window size and feature dimension traces a recall-memory frontier.",
      "1.3B models", ["based"], status="supported", supports=[["def.recall", "t.linear_attn_is_rnn"]]),
    n("e.hybrid_recovers", "empirical_claim",
      "A little attention recovers most of the recall gap: input-dependent sparse-attention hybrids close 97.4% of "
      "the Pile perplexity gap, and a 125M H3+attention hybrid beats a Transformer by 1.0 perplexity on OpenWebText.",
      "70M-360M (Zoology), 125M (H3)", ["zoology", "h3"], status="supported", supports=[["e.recall_gap"]]),
    n("e.gla_parity", "empirical_claim",
      "Gated linear attention nearly matches Transformer++: WikiText perplexity 28.65 vs 28.39 at 340M (15B tokens) "
      "and 17.22 vs 16.85 at 1.3B (100B tokens).", "340M and 1.3B", ["gla"], status="supported",
      supports=[["t.linear_attn_is_rnn"]]),
    n("e.delta_rule", "empirical_claim",
      "Delta-rule updates beat additive linear attention on WikiText-103 at about 40M and 90M parameters; pure linear "
      "models still underperform Transformers on in-context retrieval at 1.3B; gated delta rules outperform Mamba2 "
      "and DeltaNet.", "about 40M-1.3B", ["schlag", "deltanet", "gdeltanet"], status="supported",
      supports=[["t.fast_weights"]]),
    n("e.ssm_lm", "empirical_claim",
      "Selective state-space models match same-size Transformers on language modeling at billion scale; Mamba-2 "
      "trains 2-8x faster than Mamba while remaining competitive.", "3B (Mamba), 8B (empirical study)",
      ["mamba", "mamba2", "waleffe"], status="supported"),
    n("e.looped_match", "empirical_claim",
      "A k-layer block looped L times nearly matches a kL-layer model on synthetic reasoning and is competitive on "
      "downstream reasoning; looped transformers fit in-context learning tasks with under 10% of the parameters.",
      "synthetic tasks and LM reasoning; sizes not stated in abstracts", ["saunshi", "yanglooped"],
      status="supported", notes="Motivated by t.depth_bound; the evidence stands on its own experiments."),
    n("e.recursion_flops", "empirical_claim",
      "A shared recursive block with learned per-token recursion depth (Mixture-of-Recursions) lowers validation "
      "perplexity at equal training FLOPs with fewer parameters.", "135M-1.7B, compute-matched", ["mor"],
      status="supported", notes="Motivated by t.uniform_compute."),
    n("e.adaptive_depth_lm", "empirical_claim",
      "Learned adaptive computation (ACT) gives large gains on synthetic algorithmic tasks but small gains on "
      "character-level language modeling; adaptive per-layer looping mainly helps math reasoning, with gated memory "
      "recovering commonsense performance.", "ACT: char-level LM; adaptive loops: 2026 workshop preprint",
      ["act", "adaptiveloops"], status="supported", notes="Motivated by t.depth_bound."),
    n("e.mod", "empirical_claim",
      "Routing a capacity-limited subset of tokens through each block (Mixture-of-Depths) matches baseline quality at "
      "equivalent training FLOPs and wall-clock, with faster steps at sampling.", "sizes not stated in abstract",
      ["mod"], status="supported", notes="Motivated by t.uniform_compute."),
    n("e.early_exit", "empirical_claim",
      "Per-token early exit and self-speculative decoding cut inference compute (roughly 2-3x) while keeping quality.",
      "inference-time savings on Llama-family models; not rewarded by this study's training-FLOP budget",
      ["calm", "layerskip"], status="supported", notes="Motivated by t.uniform_compute."),
    n("e.memory_tokens", "empirical_claim",
      "Learnable memory tokens improve translation and language modeling; memory tokens passed between segments make "
      "a recurrent model on par with Transformer-XL; a 3.2M single-block UT with ACT reaches non-trivial "
      "Sudoku-Extreme accuracy only with memory tokens.", "mixed; the Sudoku result is one task and one preprint",
      ["burtsev", "rmt", "utmemory"], status="supported"),
    n("e.residual_variants", "empirical_claim",
      "Depth-weighted averaging of earlier block outputs (DenseFormer) improves perplexity with a few thousand extra "
      "parameters; multiple residual streams with learned connections (hyper-connections) report significant gains "
      "over residual connections; constraining them to a manifold restores the identity mapping.",
      "author-reported; model sizes not stated in abstracts", ["denseformer", "hc", "mhc", "ablaterescue"],
      status="supported"),
    n("e.sharing", "empirical_claim",
      "Immediate block-wise weight sharing improves accuracy at no parameter cost (+0.7% and +0.8% at 125M and 350M); "
      "cross-layer sharing cuts parameters substantially in BERT-scale encoders.",
      "125M-350M decoders; BERT-scale encoders", ["mobilellm", "albert", "takase"], status="supported"),
    n("e.shape", "empirical_claim",
      "Evidence on depth versus width conflicts: over a wide range shape matters little relative to parameter count "
      "(Kaplan); at fixed parameters deeper models generalize more compositionally (41M-374M); deep-and-thin is "
      "chosen at 125M-350M.", "41M-1B+; nothing at 5-30M", ["kaplan", "petty", "mobilellm"], status="disputed"),
    n("e.tinystories", "empirical_claim",
      "Models under 10M parameters trained on TinyStories produce coherent English.", "TinyStories corpus",
      ["tinystories"], status="supported"),
    n("e.moe_granularity", "empirical_claim",
      "Fine-grained experts beat feed-forward-sized experts across compute budgets, and the dense-versus-MoE "
      "efficiency gap widens with scale.", "about 129M-3.7B on C4; no evidence at 5-30M",
      ["finemoe", "switch"], status="supported"),
    n("e.glu", "empirical_claim",
      "GLU-family feed-forward layers (SwiGLU, GEGLU) improve over ReLU and GELU feed-forward layers.",
      "T5-style models", ["glu"], status="supported"),
    n("e.prenorm", "empirical_claim",
      "Pre-norm removes the learning-rate warmup that post-norm needs; RMSNorm matches LayerNorm and is cheaper.",
      "translation and pretraining experiments", ["xiong", "rmsnorm"], status="supported"),
    n("e.instability", "empirical_claim",
      "Large-scale instabilities (attention-logit growth, output-logit divergence) reproduce in small models at high "
      "learning rates, and QK-normalization and z-loss mitigate them.", "small models", ["wortsman", "qknorm"],
      status="supported"),
    n("e.position", "empirical_claim",
      "Causal LMs without positional encoding remain competitive and learn position implicitly; no positional "
      "encoding length-generalizes better than RoPE or ALiBi on downstream reasoning; ALiBi extrapolates beyond its "
      "training length.", "various; ALiBi at 1.3B", ["haviv", "kazemnejad", "alibi"], status="supported"),
    n("e.tied_small", "empirical_claim",
      "At small scale embeddings are a large share of parameters (18% of the 5.9M baseline at vocabulary 4096), and "
      "tying is standard in small-model recipes.", "this harness; 125M-350M recipes", ["harness", "mobilellm",
                                                                                       "presswolf"],
      status="supported"),
    n("e.proxy_limits", "empirical_claim",
      "Average perplexity can hide recall deficits, and simple synthetic recall tests can miss language-model "
      "behavior; MAD's synthetic tasks are reported to predict compute-optimal perplexity.", "70M-7B (MAD)",
      ["zoology", "mad"], status="disputed"),

    # ---------------- known alternatives (background, not active) ----------------
    n("alt.ut", "method", "Universal Transformer: one shared block applied recurrently over depth with per-position "
      "dynamic halting.", "algorithmic tasks, LAMBADA, WMT", ["ut", "act"],
      supports=[["e.adaptive_depth_lm"]],
      notes="Motivated by t.depth_bound."),
    n("alt.pondernet", "method", "PonderNet: probabilistic halting that learns the number of steps end to end.",
      "synthetic tasks, QA", ["pondernet"], supports=[["e.adaptive_depth_lm"]]),
    n("alt.recurrent_depth", "method", "Recurrent-depth LM: prelude, a core block iterated a variable number of "
      "times, and coda; test-time compute scales in latent space.", "3.5B", ["geiping", "saunshi", "giannou"],
      supports=[["e.looped_match"]]),
    n("alt.mor", "method", "Mixture-of-Recursions: shared recursive block with per-token routing to different "
      "recursion depths.", "135M-1.7B", ["mor", "relaxedrec"], supports=[["e.recursion_flops"]]),
    n("alt.mod", "method", "Mixture-of-Depths: a router sends only top-k tokens through each block; others skip it "
      "on the residual stream.", "see e.mod", ["mod"], supports=[["e.mod"]]),
    n("alt.early_exit", "method", "Early exit: tokens stop at an intermediate layer once confident.",
      "inference", ["calm", "layerskip"], supports=[["e.early_exit"]]),
    n("alt.linear_attention", "method", "Linear or gated linear attention: a fixed-size matrix state replaces the KV "
      "history; log-linear variants grow the state logarithmically.", "see e.gla_parity",
      ["katharopoulos", "gla", "loglinear"], supports=[["t.linear_attn_is_rnn", "e.gla_parity"]]),
    n("alt.delta_rule", "method", "Delta-rule state updates (DeltaNet, Gated DeltaNet, RWKV-7): the state is "
      "corrected toward new key-value pairs instead of summed.", "see e.delta_rule",
      ["deltanet", "gdeltanet", "rwkv7"], supports=[["t.fast_weights", "e.delta_rule"]]),
    n("alt.ssm", "method", "State-space sequence mixing (S4, Mamba, Mamba-2): input-dependent linear recurrence "
      "with a fixed-size state.", "see e.ssm_lm", ["s4", "mamba", "mamba2"], supports=[["e.ssm_lm"]]),
    n("alt.retention", "method", "Retention and RWKV-style recurrences with parallel training and recurrent "
      "inference.", "language modeling up to 14B", ["retnet", "rwkv"], supports=[["t.linear_attn_is_rnn"]]),
    n("alt.local_hybrid", "method", "Local (sliding-window) attention combined with a recurrent or linear-attention "
      "layer (Griffin, Samba, BASED, Jamba).", "125M-14B", ["longformer", "griffin", "samba", "based", "jamba"],
      supports=[["e.hybrid_recovers", "e.recall_state"]]),
    n("alt.memory_tokens", "method", "Memory or register tokens: extra latent positions that read and write global "
      "state.", "see e.memory_tokens", ["burtsev", "rmt", "registers"], supports=[["e.memory_tokens"]]),
    n("alt.multi_stream", "method", "Hyper-connections: several residual streams with learned depth and width "
      "connections.", "see e.residual_variants", ["hc", "mhc"], supports=[["e.residual_variants"]]),
    n("alt.depth_mixing", "method", "Learned access to earlier layers: depth-weighted averaging (DenseFormer) or "
      "gated highway paths.", "see e.residual_variants", ["denseformer", "highway"],
      supports=[["e.residual_variants"]]),
    n("alt.layer_sharing", "method", "Cross-layer or block-wise weight sharing.", "see e.sharing",
      ["albert", "mobilellm", "takase"], supports=[["e.sharing"]]),
    n("alt.moe", "method", "Sparse mixture-of-experts feed-forward layers with learned routing.",
      "see e.moe_granularity", ["moe", "switch", "olmoe"], supports=[["e.moe_granularity"]]),
    n("alt.position", "method", "ALiBi distance biases or no positional encoding (NoPE).", "see e.position",
      ["alibi", "haviv"], supports=[["e.position"]]),
    n("alt.qk_norm", "method", "QK-normalization (and z-loss) for attention-logit stability.", "see e.instability",
      ["qknorm", "wortsman"], supports=[["e.instability"]]),
]

space = dict(
    id="lm-state-compute",
    version=0,
    parent_version=None,
    context=dict(
        fixed_requirements=[
            "Next-token prediction on the frozen task corpus with its fixed tokenizer (vocabulary 4096), context "
            "length 512, and batch 32",
            "Causal prefix access only: the prediction at t depends only on tokens <= t of the same sequence",
            "No state carried across sequences, batch rows, or evaluation calls",
            "Equal training-FLOP budget per candidate, counted as executed, with the same data order and init seed "
            "for paired comparison",
            "Evaluator, data splits, and engineering checks are fixed; the confirm split is sealed",
            "At most 60M parameters",
        ],
        editable_priors=[
            "Matched training FLOPs is the fair comparison (rather than matched parameters or wall-clock)",
            "Early-checkpoint dev loss is informative about longer-run dev loss (Stage A tests this)",
            "Results reported at 70M-8B transfer qualitatively to 5-30M models",
            "Average next-token loss is the primary utility; mechanism-specific probes are diagnostics",
        ],
        permitted_constructions=[
            "Any torch modules, including custom recurrences over depth or sequence within one forward pass",
            "Per-token adaptive computation, with an auxiliary loss returned as aux_loss",
            "Parameter sharing, routing, gating, extra latent positions, or new state variables within a sequence",
            "Recipe changes within the allowed train_config bounds",
        ],
    ),
    nodes=nodes,
)

validate_space(space)
out = Path(__file__).resolve().parent / "lm_state_compute_v0.json"
out.write_text(json.dumps(space, indent=2, ensure_ascii=False), encoding="utf-8")
by_type = {}
for x in nodes:
    by_type[x["type"]] = by_type.get(x["type"], 0) + 1
print(f"wrote {out}: {len(nodes)} nodes {by_type}, {sum(x['active'] for x in nodes)} active, "
      f"{len({s for x in nodes for s in x['sources']})} distinct sources")
