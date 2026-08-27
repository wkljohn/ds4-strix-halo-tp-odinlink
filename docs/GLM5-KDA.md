# GLM-5.3 resident KDA component status

This research branch validates the resident linear-attention component needed
by GLM-5.3-Flash. It does **not** yet provide usable GLM-5.3 inference.

The attention schedule is derived from GGUF tensor presence rather than layer
number. For the currently validated 46-layer GGUF, 34 KDA layers require
152,633,344 bytes (145.56 MiB) of resident per-sequence state on each TP rank.
The state contains three causal-convolution histories and the recurrent matrix
for every KDA layer. Both ranks compute this state independently; KDA performs
no TP exchange.

No expanded-weight cache is created. Projections consume the compact GGUF
weights through the existing model-range mapping, while temporary activation
storage is bounded by the selected token chunk. This component does not enable
MTP/DSpark or vision.

Run the complete component gate with the workspace-pinned ROCm 7.14 build:

```bash
export DS4_RESEARCH_ROOT=/home/wkljohn/Desktop/cc/research-results
export DS4_GLM5_MODEL=/absolute/path/GLM-5.3-Flash-Q4_K.gguf
make -j"$(nproc)" strix-halo
make test-glm5-resident-kda
```

The gate checks strict metadata/tensor binding, same-GGUF projection and full
layer references, causal convolution, recurrence, prefill-to-decode handoff at
1 through 2,048 tokens, independent-rank digests, corruption rejection, TP
feature negotiation, and the explicit incomplete-graph refusal.

Remaining graph work includes sparse NoPE MLA and its indexer/cache, mHC
stream mixing, dense and routed FFNs, sharded top-8 MoE composition, embedding
and output stages, session lifecycle, and complete mandatory-RDMA TP=2
validation. Component hashes are diagnostic identifiers, not promoted model
quality references.
