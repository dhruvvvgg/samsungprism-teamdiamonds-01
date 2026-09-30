# int8 Precision Evaluation Console Output

The JSON artifact `cpu_precision.json` from the evaluation run was not retained. The raw console measurement output is transcribed exactly below:

```text
fp32: 7994 ms/query (p50), peak RSS 11023.6 MB, load 13.96s | int8: 5976 ms/query (p50), peak RSS 11104.3 MB, load 25.31s | cos to fp32 0.068025 (min 0.025332), top-10 overlap 0.0100, rank-1 changed 19 (95.0%) | census: Linear layers fp32 196 -> 0, quantised 0 -> 392
```

### Analysis & Verdict
- **Quality Degradation**: While dynamic int8 quantization reduced median latency from 7,994 ms to 5,976 ms on CPU, it degraded retrieval fidelity catastrophically (cosine similarity to fp32 dropped to 0.068, top-10 overlap dropped to 1%, and rank-1 changed on 19 of 20 queries).
- **Census Double-Count**: The census line `quantised 0 -> 392` reported double the actual 196 Linear layers because PyTorch's `named_modules()` traversal visited both each dynamically-quantized `Linear` layer and its internal `LinearPackedParams` submodule. This bookkeeping issue has been fixed in `src/retrieval/dense_encoder.py`.
- **Verdict**: int8 is **rejected** for serving and retrieval.
