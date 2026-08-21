# Antirez Q4_K RDMA validation — 2026-08-21

## Scope

Fixed `ds4-bench-tp` comparison of the following model over the two mandatory
RDMA providers:

```text
/home/wkljohn/Desktop/cc/models/antirez-deepseek-v4-gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf
```

- Model size: `164633502592` bytes
- Model sample SHA-256: `73ee63957de60fcd65c0eadb6bf826de95ac45ca2a26fd42a08917aa75ff3cc1`
- `ds4` SHA-256: `f44b229289d191d0ef5f8b9fec75a4ee428c8af53dc2bb8f1e4ff21a37d5f882`
- `ds4-bench-tp` SHA-256: `ad2ab9c8639591d74cccd914f7c1e3f39806cbd40d8f25e85a0909dd000e0273`
- Workload: 2,048 prompt tokens followed by 300 generated tokens
- Expert split: balanced 128/128
- Persistent expanded-weight cache: disabled
- Token fingerprint: `b7694f9d11a3760e`

Both nodes used independent filesystems with identical model paths and sampled
model fingerprints. Every run passed the launcher's explicit transport checks;
provider fallback or a token-count/fingerprint mismatch would have failed the
run.

## Results

| Provider | Run 1 | Run 2 | Run 3 | Median prefill | Median decode |
|---|---:|---:|---:|---:|---:|
| RoCE v2 | 273.70 / 19.54 | 272.51 / 19.70 | 271.42 / 19.44 | **272.51 t/s** | **19.54 t/s** |
| OdinLink | 231.54 / 19.19 | 231.46 / 19.13 | 231.34 / 19.03 | **231.46 t/s** | **19.13 t/s** |

Each run cell is `prefill t/s / decode t/s`. The historical baseline was not
remeasured and remains unchanged in the public table.

Raw CSV, coordinator logs, worker logs, and manifests are retained in the
git-ignored directory:

```text
research-results/decode-25tps-2026-08-20/runs/antirez-q4-{roce,odinlink}-r{1,2,3}-20260821.*
```
