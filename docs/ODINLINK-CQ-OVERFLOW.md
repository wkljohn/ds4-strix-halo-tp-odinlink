# OdinLink CQ ring overflows on ds4's bulk path (found offline, NOT yet fixed)

A latent hang on the ds4 TP bulk path, pinned to exact numbers. **No OdinLink
code has been changed** - the installed shim is what the working vLLM/RCCL
stack uses, so replacing it is a decision for the user, not a side effect of
this work.

## The mismatch

ds4 asks for a 512-entry completion queue (`ds4_tp.c:742`):

    r->cq = r->api.create_cq(r->ctx, 512, NULL, NULL, 0);

OdinLink accepts it and **reports 512 back** (`odl_tb5_verbs_cq.c:36`):

    cq->base.cqe = cqe > 0 ? cqe : 1;

but the completions land in a **fixed-size array** that ignores `cqe`
(`odl_tb5_verbs.h:132`):

    struct ibv_wc ring[ODL_VERBS_COMP_CHANNEL_BACKLOG];   /* = 64 */

so real capacity is **63** (full when `(tail+1) % 64 == head`), while
`ibv_create_cq` advertises 512. The provider lies about its depth.

On overflow the completion is **dropped** (`odl_tb5_verbs_cq.c:177-183`):

    int next = (cq->tail + 1) % ODL_VERBS_COMP_CHANNEL_BACKLOG;
    if (next == cq->head) {
        odl_logerr("CQ %p ring full! dropping completion", (void*)cq);
        return -ENOSPC;
    }

## Why 63 is not enough

One CQ serves both directions (`ds4_tp.c:748-749`: `send_cq = recv_cq = r->cq`).

Per bulk round (`ds4_tp.c:1150-1232`), with
`DS4_TP_RDMA_BULK_SLOTS = 64`, `DS4_TP_RDMA_MAX_MSG = 16384`:

| source | count | note |
|---|---|---|
| recv completions | up to **64** | `ibv_post_recv` of `chunks` WRs; receives are always signaled |
| send completions | **1** | only the last WR sets `IBV_SEND_SIGNALED` (`:1209`) |
| **worst case in flight** | **65** | vs **63** usable |

ds4 itself expects 65 - it polls for exactly that:

    struct ibv_wc wc[DS4_TP_RDMA_BULK_SLOTS + 1u];
    int n = ibv_poll_cq(r->cq, (int)(DS4_TP_RDMA_BULK_SLOTS + 1u), wc);

If the peer's RX thread posts faster than ds4 drains, 2 completions are dropped,
and the loop

    while (recv_done < chunks || !send_done)

can never satisfy its condition. It spins to `tp->timeout_sec` and the transfer
fails. **Load-dependent, so it may pass a light smoke test and hang under a real
prefill.**

`chunks` reaches 64 whenever `remaining >= 64 * 16 KB = 1 MB`, which the big
gate exchange (2 MB rounds) does on its first round.

## Not a decode-path risk

Decode uses the recv window, not bulk (`ds4_tp.c:1040-1043`):

    chunks_per_gate = ceil(vec_bytes / 16384)
    nwr = DS4_TP_RDMA_RECV_WINDOW * chunks_per_gate     /* RECV_WINDOW = 16 */

For this model `vec_bytes = DS4_N_EMBD * 4 = 4096 * 4 = 16384` exactly, so
`chunks_per_gate = 1` and `nwr = 16`. Comfortably inside 63. **Decode is safe;
prefill/bulk is the exposure.**

## Options (user decision - touches the working RCCL shim)

1. **Bump the constant** to >= 512 in `odl_tb5_verbs.h:78`. One line. Costs
   `512 * sizeof(struct ibv_wc)` ~= 24 KB per CQ. Smallest change, but keeps the
   lie - a consumer asking for 4096 would still silently get 512.
2. **Honour `cqe`**: allocate the ring from the requested depth. Correct, and
   removes the class of bug rather than this instance. Changes the struct size,
   so every component linking the shim must be rebuilt together.
3. **Clamp honestly**: keep 64 but report `base.cqe = min(cqe, 63)`. ds4 would
   then see the real depth - though ds4 does not currently read it back, so this
   alone does not stop the drop.

Recommendation: **(2)**, with (1) as the low-risk stopgap if the RCCL stack must
not be disturbed. Either way the fix belongs behind the same patch strategy as
the ds4 changes, built to a separate artefact, with the working shim left in
place until the user approves the swap.
