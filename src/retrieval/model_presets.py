"""Per-model settings for run_baseline.py --preset NAME. Keys map to run_baseline's CLI arguments.

Existing models (MiniLM, CodeRankEmbed, gte-modernbert, Qwen3) are still driven by explicit CLI flags;
presets only exist for models whose config is easy to get wrong.
"""

F2LLM_INSTRUCTION = "Retrieve the most relevant code snippet for the given query."

# Informational only (kept OUT of PRESETS so apply_preset never sets these as CLI args). Values read
# from the installed MTEB registry's ModelMeta entries (codefuse_models.py): memory_usage_mb is MTEB's
# own recorded footprint for the model, n_parameters its parameter count. Used by dev_split.py to warn
# before a run that would not fit the GPU. A T4 has 14.56 GB total.
PRESET_MEMORY = {
    "f2llm-v2-0.6b": {"registry_memory_mb": 2274, "n_parameters": 596_049_920},
    "f2llm-v2-1.7b": {"registry_memory_mb": 6563, "n_parameters": 1_720_574_976},
    "f2llm-v2-4b": {"registry_memory_mb": 15344, "n_parameters": 4_022_468_096},
}
T4_TOTAL_MB = 14560


def memory_warning(preset_name, warn_threshold_mb=13000, gpu_total_mb=T4_TOTAL_MB):
    """Return a human-readable warning string if this preset's registered memory footprint is at or over
    `warn_threshold_mb`, else None. Informational: callers warn, they do not refuse to run."""
    info = PRESET_MEMORY.get(preset_name)
    if not info or info["registry_memory_mb"] < warn_threshold_mb:
        return None
    mb, params = info["registry_memory_mb"], info["n_parameters"]
    fp16_weights_mb = params * 2 / 1e6
    over = "EXCEEDS" if mb > gpu_total_mb else "is within but close to"
    return (
        f"preset {preset_name!r} has registry memory_usage_mb={mb} (~{mb / 1000:.1f} GB), which {over} "
        f"the GPU's {gpu_total_mb / 1000:.2f} GB total and is at/over the {warn_threshold_mb / 1000:.1f} GB "
        f"warn threshold.\n"
        f"  Its {params / 1e9:.2f}B parameters are ~{fp16_weights_mb / 1000:.1f} GB of fp16 weights alone, "
        f"before any activations. Weight memory does NOT shrink with --batch-size; only activations do, so "
        f"a smaller batch may or may not be enough.\n"
        f"  If this OOMs: lower --batch-size first (activations only), then consider a smaller preset."
    )

PRESETS = {
    # Sources (read, not run):
    #  - MTEB registry: mteb/models/model_implementations/codefuse_models.py -> F2LLM_v2_0B6
    #    (loader=InstructSentenceTransformerModel, instruction_template="Instruct: {instruction}\nQuery: ",
    #     apply_instruction_to_passages=False, max_seq_length=8192, revision 54b4e2dc..., torch_dtype=BF16)
    #  - f2llmv2_prompts_dict["AppsRetrieval"] = F2LLM_INSTRUCTION
    #  - HF repo: modules.json (Transformer -> Pooling[last-token] -> Normalize), tokenizer.json
    #    post-processor appends <|im_end|> (id 151645) to every input.
    "f2llm-v2-0.6b": {
        "model": "codefuse-ai/F2LLM-v2-0.6B",
        "revision": "54b4e2dc74e01be7126d4cf5f016af6b21edc563",
        "query_prefix": f"Instruct: {F2LLM_INSTRUCTION}\nQuery: ",
        "doc_prefix": "",
        "max_seq_length": 8192,
        "dtype": "fp16",  # published run used bf16; a T4 has no native bf16, so fp16 (auto-falls back to fp32 on overflow)
        "trust_remote_code": False,
        "expect_eos": True,
    },
    # Same lineage/loader_kwargs as F2LLM_v2_0B6 in the registry (verified: identical instruction_template,
    # prompts_dict, apply_instruction_to_passages, add_eos_token, max_seq_length, torch_dtype=BF16), just a
    # bigger checkpoint -- codefuse_models.py -> F2LLM_v2_1B7, revision 3766d46e..., n_parameters 1,720,574,976
    # (0.6B's is 596,049,920), embed_dim 2048 (0.6B's is 1024). HF repo confirmed identical: modules.json
    # (Transformer -> Pooling[last-token] -> Normalize), config.json architectures=["Qwen3Model"],
    # eos_token_id 151645, no auto_map (no custom code, trust_remote_code not needed).
    # Published test NDCG@10 0.93692 / MRR@10 0.92288 at this exact revision (MTEB results repo, mteb 2.6.7).
    "f2llm-v2-1.7b": {
        "model": "codefuse-ai/F2LLM-v2-1.7B",
        "revision": "3766d46e7a68545ed6190c15330983f9b39ab718",
        "query_prefix": f"Instruct: {F2LLM_INSTRUCTION}\nQuery: ",
        "doc_prefix": "",
        "max_seq_length": 8192,
        "dtype": "fp16",  # published run used bf16; a T4 has no native bf16, so fp16 (auto-falls back to fp32 on overflow)
        "trust_remote_code": False,
        "expect_eos": True,
    },
    # Same lineage/loader_kwargs as F2LLM_v2_0B6/F2LLM_v2_1B7 (verified: identical instruction_template,
    # prompts_dict, apply_instruction_to_passages, add_eos_token, max_seq_length, torch_dtype=BF16) --
    # codefuse_models.py -> F2LLM_v2_4B, revision e04d1a04..., n_parameters 4,022,468,096, embed_dim 2560.
    # HF repo confirmed identical structure: modules.json (Transformer -> Pooling[last-token] ->
    # Normalize), config.json architectures=["Qwen3Model"], eos_token_id 151645, no auto_map.
    # Published test NDCG@10 0.96102 / MRR@10 0.95170 at this exact revision (MTEB results repo, mteb 2.6.7).
    # CAUTION: registry memory_usage_mb=15344 (~15.3 GB) -- close to or over a T4's 14.56 GB total. fp16
    # weights alone are ~8 GB (4.02B params x 2 bytes); the registry figure likely reflects a larger
    # benchmark batch size than we'd use here, but this is genuinely tight -- use a small --batch-size
    # (e.g. 2-4) for dev_split.py and watch for OOM.
    "f2llm-v2-4b": {
        "model": "codefuse-ai/F2LLM-v2-4B",
        "revision": "e04d1a04f4a0e154bf43969d3dcc596bbd92b1f8",
        "query_prefix": f"Instruct: {F2LLM_INSTRUCTION}\nQuery: ",
        "doc_prefix": "",
        "max_seq_length": 8192,
        "dtype": "fp16",  # published run used bf16; a T4 has no native bf16, so fp16 (auto-falls back to fp32 on overflow)
        "trust_remote_code": False,
        "expect_eos": True,
    },
}


def apply_preset(args, defaults, name):
    """Fill `args` (argparse Namespace) from PRESETS[name], but never override a value the user set
    explicitly (i.e. one that differs from the parser default)."""
    if name not in PRESETS:
        raise SystemExit(f"Unknown --preset {name!r}; available: {sorted(PRESETS)}")
    for key, value in PRESETS[name].items():
        if getattr(args, key, None) == defaults.get(key):
            setattr(args, key, value)
    return args
