#!/usr/bin/env python3
"""Patch a STOCK antirez/ds4 tree for tensor parallelism on AMD gfx1151.

STRATEGY: patches, never a fork. This applies to an unmodified upstream
checkout so the tree stays rebaseable and the delta stays reviewable. Same
discipline that worked for vllm-strix-dsv4:

  * IDEMPOTENT      - running twice is a no-op; every patch detects its own
                      prior application before touching anything.
  * FAIL-CLOSED     - anchors are unique strings. If upstream moves, the anchor
                      count changes and the patch REFUSES rather than guessing.
                      A silently-misapplied patch is worse than an unpatched
                      tree, because every downstream signal then says "fine".
  * --check         - dry run. Reports what WOULD change, writes nothing.
  * BACKUPS         - .orig-ds4tp beside every file touched, written once.

Upstream pinned: antirez/ds4 @ 54b36ed (2026-07-28).

    python3 patch_ds4_gfx1151_tp.py --tree /path/to/ds4 [--check]
"""
import argparse
import os
import sys

MARKER = "DS4-TP-gfx1151"
BACKUP_SUFFIX = ".orig-ds4tp"

WOULD, APPLIED, ALREADY, FAIL = "would", "applied", "already", "FAIL"
_RESULTS = []


def _log(name, state, detail):
    sym = {WOULD: " ??", APPLIED: " ->", ALREADY: " ==", FAIL: " !!"}[state]
    print(f"{sym} {name}: {state} - {detail}")
    _RESULTS.append((name, state))


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write_checked(path, old, new, check):
    """Write only if the content actually changed, and back up once.

    Never writes when `check` is set. Returns True if a write happened (or
    would have).
    """
    if old == new:
        return False
    if check:
        return True
    bak = path + BACKUP_SUFFIX
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8") as f:
            f.write(old)
    tmp = path + ".tmp-ds4tp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new)
    os.replace(tmp, path)  # atomic; a crash mid-write cannot leave a half file
    return True


# ---------------------------------------------------------------------------
# Patch 1: let the verbs RDMA path compile on Linux, not just macOS.
#
# ds4's RDMA is STANDARD libibverbs - ibv_reg_mr / ibv_post_send / ibv_poll_cq
# and friends, dlsym'd at runtime. Nothing about it is Apple-specific except
# the platform guard and the library filename. Apple simply ships a
# verbs-compatible RDMA-over-Thunderbolt provider; any other verbs provider
# (here: OdinLink's libodl_tb5-rdmav34.so) satisfies the same contract.
# ---------------------------------------------------------------------------
P1_ANCHOR = """#if defined(__APPLE__) && defined(__has_include)
#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#include <dlfcn.h>
#define DS4_TP_HAVE_VERBS 1
#endif
#endif"""

P1_NEW = """/* DS4-TP-gfx1151 (patch 1): the verbs path is not Apple-specific.
 * ds4 codes to standard libibverbs and dlsym's it at runtime; macOS just
 * happens to ship a verbs provider for Thunderbolt RDMA. On Linux the same
 * contract is satisfied by any rdma-core provider - for this deployment,
 * OdinLink's libodl_tb5-rdmav34.so over Thunderbolt 5. Widen the guard to
 * any platform that actually has <infiniband/verbs.h>. */
#if defined(__has_include)
#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#include <dlfcn.h>
#define DS4_TP_HAVE_VERBS 1
#endif
#endif"""


def patch_verbs_platform(tree, check):
    name = "verbs path on Linux (not just macOS)"
    path = os.path.join(tree, "ds4_tp.c")
    if not os.path.exists(path):
        _log(name, FAIL, f"missing {path}")
        return
    src = _read(path)
    if MARKER + " (patch 1)" in src:
        _log(name, ALREADY, "guard already widened")
        return
    n = src.count(P1_ANCHOR)
    if n != 1:
        _log(name, FAIL, f"anchor count {n} != 1 - upstream moved, refusing")
        return
    if _write_checked(path, src, src.replace(P1_ANCHOR, P1_NEW), check):
        _log(name, WOULD if check else APPLIED,
             "#if defined(__APPLE__) -> any platform with <infiniband/verbs.h>")


# ---------------------------------------------------------------------------
# Patch 2: load the Linux verbs library, not the macOS one.
#
# Upstream dlopens librdma.dylib (macOS). On Linux the equivalent is
# libibverbs.so.1. Try Linux names FIRST so a Linux box never depends on a
# .dylib being present, then fall through to the macOS names so this patch is
# a no-op on Apple hardware.
#
# WHICH verbs library to load - this was WRONG in the first version.
#
# I originally loaded the SYSTEM libibverbs, reasoning that OdinLink's shim was
# incomplete. Two things falsified that:
#
#  1. The rdma-core PROVIDER path cannot work at all here. The provider only
#     calls verbs_register_driver_34 with no match_table/match_device, so its
#     callbacks fire only after libibverbs has already discovered a KERNEL
#     device. odl_tb5.ko registers a char device and never calls
#     ib_register_device, so /sys/class/infiniband is empty and enumeration
#     yields nothing. Verified: `IBV_CONFIG_DIR=... ibv_devices` -> empty.
#
#  2. The shim is NOT incomplete - the copy I inspected was two days STALE.
#     The current build exports ibv_get_device_list and ibv_query_gid (it
#     synthesises the device list and forwards everything else via RTLD_NEXT),
#     and covers 100% of the symbols this function resolves. With it preloaded,
#     `ibv_devices` lists odl_tb5_0 and `ibv_devinfo` reports PORT_ACTIVE.
#
# And critically: dlsym() against a handle from dlopen("libibverbs.so.1")
# resolves in the SYSTEM library, so LD_PRELOAD interposition is BYPASSED. The
# shim must be opened explicitly. DS4_TP_VERBS_LIB overrides the path.
# ---------------------------------------------------------------------------
P2_ANCHOR = """    void *h = dlopen("/usr/lib/librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);"""

P2_NEW = """    /* DS4-TP-gfx1151 (patch 2): pick the verbs library explicitly.
     * dlsym() on a handle from dlopen("libibverbs.so.1") resolves inside the
     * SYSTEM library, so an LD_PRELOAD'd interposing shim would be bypassed -
     * the OdinLink shim must be opened by name. It synthesises the device list
     * for Thunderbolt RDMA and forwards everything else via RTLD_NEXT, so it is
     * a superset, not a replacement. Falls through to the system library, then
     * to macOS names, so this is a no-op off Linux/OdinLink. */
    void *h = NULL;
    const char *tp_verbs_lib = getenv("DS4_TP_VERBS_LIB");
    if (tp_verbs_lib) h = dlopen(tp_verbs_lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libodl_tb5_verbs.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libibverbs.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("/usr/lib/librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);"""


def patch_verbs_dlopen(tree, check):
    name = "dlopen libibverbs.so.1 on Linux"
    path = os.path.join(tree, "ds4_tp.c")
    if not os.path.exists(path):
        _log(name, FAIL, f"missing {path}")
        return
    src = _read(path)
    if MARKER + " (patch 2)" in src:
        _log(name, ALREADY, "Linux verbs library already preferred")
        return
    n = src.count(P2_ANCHOR)
    if n != 1:
        _log(name, FAIL, f"anchor count {n} != 1 - upstream moved, refusing")
        return
    if _write_checked(path, src, src.replace(P2_ANCHOR, P2_NEW), check):
        _log(name, WOULD if check else APPLIED,
             "librdma.dylib -> libibverbs.so.1 first, .dylib retained as fallback")


# ---------------------------------------------------------------------------
# Patch 3: let a ROCm build accept TP options - but ONLY when the ROCm TP
# runtime is actually compiled in.
#
# Upstream hard-gates TP to Metal (ds4_tp.c:504-507). Simply deleting that
# check is UNSAFE: the ROCm gate entry points are still stubs that print
# "tensor parallelism is Metal-only" and return 0 (ds4_rocm.cu:137-215), so
# TP would be accepted at the CLI and then silently do nothing at runtime -
# the exact "healthy but silently degraded" failure this project has been
# bitten by before.
#
# So the gate is widened only under DS4_ROCM_TP_READY, which patch 4 (the
# ROCm gate runtime) defines. Until then a ROCm build still refuses TP, loudly
# and at option-parse time, which is the correct fail-closed behaviour.
# ---------------------------------------------------------------------------
P3_ANCHOR = """    if (opt->backend != DS4_BACKEND_METAL) {
        tp_set_err(err, errlen, "tensor parallelism requires the Metal backend");
        return 0;
    }"""

P3_NEW = """    /* DS4-TP-gfx1151 (patch 3): allow ROCm once its TP runtime exists.
     * Deliberately gated on DS4_ROCM_TP_READY (defined by the ROCm gate-runtime
     * patch), NOT on the backend alone: the ROCm gate entry points ship as
     * stubs, so accepting TP before they are real would give a CLI that says
     * yes and a runtime that does nothing. Fail closed until proven. */
#ifdef DS4_ROCM_TP_READY
    /* NOTE there is no DS4_BACKEND_ROCM. ds4.h:20-22 defines only METAL, CUDA
     * and CPU - a ROCm build IS the CUDA backend, HIP-translated (hence the
     * cuda*->hip* macro wall in ds4_rocm.h). So the widened test names
     * DS4_BACKEND_CUDA. That is still fail-closed for a genuine CUDA build,
     * because stock CUDA also stubs the gate encoders
     * (ds4_cuda.cu:27319,27324 "CUDA stub called") and therefore never
     * defines DS4_ROCM_TP_READY. */
    if (opt->backend != DS4_BACKEND_METAL && opt->backend != DS4_BACKEND_CUDA) {
        tp_set_err(err, errlen,
                   "tensor parallelism requires the Metal or ROCm backend");
        return 0;
    }
#else
    if (opt->backend != DS4_BACKEND_METAL) {
        tp_set_err(err, errlen, "tensor parallelism requires the Metal backend");
        return 0;
    }
#endif"""


def patch_tp_backend_gate(tree, check):
    name = "accept ROCm backend for TP (gated on DS4_ROCM_TP_READY)"
    path = os.path.join(tree, "ds4_tp.c")
    if not os.path.exists(path):
        _log(name, FAIL, f"missing {path}")
        return
    src = _read(path)
    if MARKER + " (patch 3)" in src:
        _log(name, ALREADY, "ROCm gate already present")
        return
    n = src.count(P3_ANCHOR)
    if n != 1:
        _log(name, FAIL, f"anchor count {n} != 1 - upstream moved, refusing")
        return
    if _write_checked(path, src, src.replace(P3_ANCHOR, P3_NEW), check):
        _log(name, WOULD if check else APPLIED,
             "Metal-only -> Metal|CUDA(=ROCm build), only under DS4_ROCM_TP_READY")


# ---------------------------------------------------------------------------
# Patch 5: compile the TP ENGINE for non-Apple builds.
#
# THIS IS THE PATCH THAT MAKES PATCH 4 MATTER, and it was nearly missed.
# ds4_gpu_tp_init() is neither defined nor even REFERENCED in a ROCm build:
# `nm ds4.o` shows only `U ds4_gpu_tp_gate_encode`. The reason is that the whole
# TP engine - the exchange callbacks, the bind path, the failure check - sits
# behind `#if !defined(DS4_NO_GPU) && defined(__APPLE__)` in ds4.c.
#
# So implementing the ROCm gate runtime (patch 4) would have compiled cleanly
# and done NOTHING: nothing would ever call tp_init, and the runtime would never
# start. That is precisely the "healthy but silently degraded" failure this
# project has been bitten by, and reading the call site at ds4.c:56575 does not
# reveal it - only checking linkage does.
#
# Four identical guards wrap TP code (tp_exchange callbacks, two `e->tp.active`
# blocks, and the ds4_gpu_tp_failed() check). All four are widened together;
# the count is asserted so a change upstream fails the patch instead of
# half-applying it.
# ---------------------------------------------------------------------------
P5A_ANCHOR = "#if !defined(DS4_NO_GPU) && defined(__APPLE__)"
P5A_NEW = "#if !defined(DS4_NO_GPU) && (defined(__APPLE__) || defined(DS4_ROCM_TP_READY))"
P5A_COUNT = 4

P5B_ANCHOR = "#if defined(DS4_NO_GPU) || !defined(__APPLE__)"
P5B_NEW = "#if defined(DS4_NO_GPU) || (!defined(__APPLE__) && !defined(DS4_ROCM_TP_READY))"

P5C_ANCHOR = """    if (e->backend != DS4_BACKEND_METAL) {
        snprintf(err, errlen, "tensor parallelism requires the Metal backend");
        return 0;
    }"""

P5C_NEW = """    /* DS4-TP-gfx1151 (patch 5): a ROCm build reports DS4_BACKEND_CUDA
     * (ds4.h:20-22 has no ROCM enum - ROCm IS the CUDA backend, HIP-translated).
     * Only widened under DS4_ROCM_TP_READY, so a stock CUDA build - which also
     * stubs the gate encoders at ds4_cuda.cu:27319,27324 - still refuses. */
#ifdef DS4_ROCM_TP_READY
    if (e->backend != DS4_BACKEND_METAL && e->backend != DS4_BACKEND_CUDA) {
        snprintf(err, errlen, "tensor parallelism requires the Metal or ROCm backend");
        return 0;
    }
#else
    if (e->backend != DS4_BACKEND_METAL) {
        snprintf(err, errlen, "tensor parallelism requires the Metal backend");
        return 0;
    }
#endif"""


def patch_tp_engine_platform(tree, check):
    name = "compile the TP engine on non-Apple (ds4.c guards)"
    path = os.path.join(tree, "ds4.c")
    if not os.path.exists(path):
        _log(name, FAIL, f"missing {path}")
        return
    src = _read(path)
    if MARKER + " (patch 5)" in src:
        _log(name, ALREADY, "TP engine guards already widened")
        return
    na, nb, nc = src.count(P5A_ANCHOR), src.count(P5B_ANCHOR), src.count(P5C_ANCHOR)
    if na != P5A_COUNT or nb != 1 or nc != 1:
        _log(name, FAIL,
             f"anchor counts A={na}(want {P5A_COUNT}) B={nb}(want 1) C={nc}(want 1)"
             " - upstream moved, refusing")
        return
    out = src.replace(P5A_ANCHOR, P5A_NEW).replace(P5B_ANCHOR, P5B_NEW)
    out = out.replace(P5C_ANCHOR, P5C_NEW)
    if _write_checked(path, src, out, check):
        _log(name, WOULD if check else APPLIED,
             f"{P5A_COUNT} __APPLE__ TP guards + bind path widened under DS4_ROCM_TP_READY")


# ---------------------------------------------------------------------------
# Patch 6: install the ROCm TP gate runtime, replacing the Metal-only stubs.
#
# The implementation lives in patches/rocm_tp_runtime.inc next to this script
# so it stays readable as C rather than as a Python string. It replaces the
# stub block at ds4_rocm.cu:135-180 wholesale.
#
# It defines DS4_ROCM_TP_READY, which is what un-gates patches 3 and 5. Nothing
# before this point can enable TP, by construction.
#
# NOTE ds4_gpu_model_residency_skip is deliberately NOT in the .inc - it is
# already defined in ds4_rocm.cu outside the stub block, and redefining it
# would be a duplicate symbol.
# ---------------------------------------------------------------------------
# NOTE r""" - the anchor contains the two characters backslash-n inside a C
# string literal. A normal triple-quoted string would turn that into a real
# newline and the anchor would never match (it silently counted 0 first try).
P6_ANCHOR = r"""/* Tensor-parallel gates are Metal-only; stubs keep shared graph code
 * linkable (TP option validation rejects non-Metal backends). */
extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    (void)layer; (void)gate;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    (void)fn;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    (void)suspend;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    (void)paused;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    (void)enabled;
}"""


# The remaining three stubs sit AFTER ds4_gpu_model_residency_skip (which must be
# preserved - it is unrelated to TP and defined only here), so the first anchor
# cannot span them contiguously. Remove them separately; the .inc supplies all
# three. Discovered by the build: 3 redefinition errors, 0 undefined symbols.
P6B_ANCHOR = r"""extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    (void)fn;
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                                          const ds4_gpu_tensor *out_t,
                                          ds4_gpu_tensor *in_t,
                                          uint64_t bytes) {
    (void)layer; (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return 0;
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)layer; (void)rows;
    fprintf(stderr, DS4_GPU_LOG_PREFIX "tensor parallelism is Metal-only\n");
    return 0;
}"""

P6B_NEW = "/* DS4-TP-gfx1151 (patch 4): these three now live in the TP runtime above. */"


def patch_rocm_tp_runtime(tree, check):
    name = "ROCm TP gate runtime (replaces Metal-only stubs)"
    path = os.path.join(tree, "ds4_rocm.cu")
    inc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "rocm_tp_runtime.inc")
    if not os.path.exists(path):
        _log(name, FAIL, f"missing {path}")
        return
    if not os.path.exists(inc):
        _log(name, FAIL, f"missing {inc}")
        return
    src = _read(path)
    if MARKER + " (patch 4)" in src:
        _log(name, ALREADY, "ROCm TP runtime already installed")
        return
    n = src.count(P6_ANCHOR)
    if n != 1:
        _log(name, FAIL, f"anchor count {n} != 1 - upstream moved, refusing")
        return
    nb = src.count(P6B_ANCHOR)
    if nb != 1:
        _log(name, FAIL, f"second anchor count {nb} != 1 - upstream moved, refusing")
        return
    body = _read(inc)
    out = src.replace(P6_ANCHOR, "#define DS4_ROCM_TP_READY 1\n" + body)
    out = out.replace(P6B_ANCHOR, P6B_NEW)
    if _write_checked(path, src, out, check):
        _log(name, WOULD if check else APPLIED,
             "stubs -> hipStreamWaitValue64 gate runtime; defines DS4_ROCM_TP_READY")


# ---------------------------------------------------------------------------
# Patch 7: define DS4_ROCM_TP_READY for the WHOLE BUILD, not one file.
#
# Patch 6 defines it inside ds4_rocm.cu - which is a single translation unit.
# ds4.c (the TP engine guards, patch 5) and ds4_tp.c (the option gate, patch 3)
# are separate TUs and never saw it, so they kept compiling their Metal-only
# branches. The build succeeded and the runtime was in the binary, yet TP would
# still have been refused at option-parse: "compiles clean, does nothing".
#
# Caught by checking the binary rather than the exit code:
#   strings ds4 | grep 'Metal or ROCm backend'  -> 0
#   nm ds4.o    | grep ds4_gpu_tp_init          -> 0
#
# Both patch 6's local define and this one are kept: the local one documents
# the dependency at the point of use, this one actually reaches every TU.
# ---------------------------------------------------------------------------
P7_ANCHOR = 'CFLAGS="$(CFLAGS) -DDS4_ROCM_BUILD"'
P7_NEW = 'CFLAGS="$(CFLAGS) -DDS4_ROCM_BUILD -DDS4_ROCM_TP_READY=1"'


def patch_makefile_tp_flag(tree, check):
    name = "define DS4_ROCM_TP_READY build-wide (Makefile)"
    path = os.path.join(tree, "Makefile")
    if not os.path.exists(path):
        _log(name, FAIL, f"missing {path}")
        return
    src = _read(path)
    if "DS4_ROCM_TP_READY" in src:
        _log(name, ALREADY, "flag already in the strix-halo CFLAGS")
        return
    n = src.count(P7_ANCHOR)
    if n != 1:
        _log(name, FAIL, f"anchor count {n} != 1 - upstream moved, refusing")
        return
    if _write_checked(path, src, src.replace(P7_ANCHOR, P7_NEW), check):
        _log(name, WOULD if check else APPLIED,
             "strix-halo CFLAGS now carry -DDS4_ROCM_TP_READY=1")


# ---------------------------------------------------------------------------
# Patch 8: fall back from UC to RC queue pairs.
#
# HARD INCOMPATIBILITY, found before first connect rather than during it:
#   ds4_tp.c:750                 qia.qp_type = IBV_QPT_UC;
#   odl_tb5_verbs_qp.c:250-251   if (attr->qp_type != IBV_QPT_RC) {
#                                    odl_logerr("unsupported QP type: %d", ...)
#
# ds4 asks for Unreliable Connected; OdinLink's provider accepts only Reliable
# Connected. create_qp would simply fail and TP would never connect.
#
# RC is safe here: ds4 relies on the QP being CONNECTED and IN-ORDER (see the
# "UC is in-order" comment near the completion reaper), and RC provides both,
# plus acks and retransmission. The cost is a little per-message overhead, not
# a semantic change. UC is still attempted FIRST so this is a no-op against
# Apple's provider, which does support it.
# ---------------------------------------------------------------------------
P8_ANCHOR = """    qia.qp_type = IBV_QPT_UC;"""

P8_NEW = """    /* DS4-TP-gfx1151 (patch 8): try UC, fall back to RC. OdinLink's verbs
     * provider rejects everything except IBV_QPT_RC
     * (odl_tb5_verbs_qp.c:250). RC keeps the connected, in-order semantics
     * this transport depends on and merely adds reliability. */
    qia.qp_type = IBV_QPT_UC;"""

P8B_ANCHOR = """    r->qp = r->api.create_qp(r->pd, &qia);
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp(UC): %s", strerror(errno));
        return 0;
    }"""

# r""" again: the fprintf carries the two characters backslash-n. A plain
# triple-quoted string turns that into a real newline and breaks the C
# string literal - the same mistake made with P6_ANCHOR.
P8B_NEW = r"""    r->qp = r->api.create_qp(r->pd, &qia);
    if (!r->qp) {
        /* DS4-TP-gfx1151 (patch 8): providers that support only RC. */
        qia.qp_type = IBV_QPT_RC;
        r->qp = r->api.create_qp(r->pd, &qia);
        if (r->qp) {
            fprintf(stderr, "ds4: tp rdma: provider rejected UC, using RC\n");
        }
    }
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp(UC and RC): %s", strerror(errno));
        return 0;
    }"""


def patch_qp_type_fallback(tree, check):
    name = "QP type UC -> RC fallback (OdinLink accepts RC only)"
    path = os.path.join(tree, "ds4_tp.c")
    if not os.path.exists(path):
        _log(name, FAIL, f"missing {path}")
        return
    src = _read(path)
    if MARKER + " (patch 8)" in src:
        _log(name, ALREADY, "RC fallback already present")
        return
    na, nb = src.count(P8_ANCHOR), src.count(P8B_ANCHOR)
    if na != 1 or nb != 1:
        _log(name, FAIL, f"anchor counts A={na} B={nb} (want 1,1) - refusing")
        return
    out = src.replace(P8_ANCHOR, P8_NEW).replace(P8B_ANCHOR, P8B_NEW)
    if _write_checked(path, src, out, check):
        _log(name, WOULD if check else APPLIED,
             "create_qp retries with IBV_QPT_RC when UC is rejected")


# ---------------------------------------------------------------------------
# Patch 9: apply the expert-shard weight mask in routed_moe_one.
#
# The mask lives in the TP runtime (patch 4/6), but ds4_rocm.cu includes
# rocm/ds4_rocm_moe_launch.cuh BEFORE that runtime, so the call site needs a
# forward declaration.
#
# n_tok is 1 at this entry point, so the scratch buffer is n_expert floats - 6
# for DS4-Flash. The masked tensor is a struct copy with a swapped ptr, the
# same pattern ds4_cuda.cu:27498-27501 uses for x_slice.
#
# Only routed_moe_ONE is patched. The BATCH entry point serves prefill/verify;
# the mask there is the same one-liner but n_tok varies, and prefill is
# separately blocked by unavailable-stub kernels, so it is deferred rather than
# written blind.
# ---------------------------------------------------------------------------
P9_ANCHOR = "".join([
    "    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,\n",
    "                             gate_offset, up_offset, down_offset,\n",
    "                             gate_type, down_type,\n",
    "                             gate_expert_bytes, gate_row_bytes,\n",
    "                             down_expert_bytes, down_row_bytes,\n",
    "                             expert_in_dim, expert_mid_dim, out_dim,\n",
    "                             selected, weights, n_total_expert, n_expert, clamp, x, layer_index, 1,\n",
    "                             force_resident);\n}",
])

P9_NEW = "".join([
    "    /* DS4-TP-gfx1151 (patch 9): zero the routing weight of experts this rank\n",
    "     * does not own, so the two ranks' partial sums recombine to exactly the\n",
    "     * unsharded result instead of double-counting. The runtime explains why\n",
    "     * this is exact, and what it does NOT buy (no memory/compute saving). */\n",
    "    ds4_gpu_tensor masked_weights;\n",
    "    const ds4_gpu_tensor *use_weights = weights;\n",
    "    if (ds4_gpu_tp_expert_shard_active() && weights && selected && n_expert) {\n",
    "        float *scratch = (float *)cuda_tmp_alloc((uint64_t)n_expert * sizeof(float),\n",
    "                                                 \"tp expert weight mask\");\n",
    "        if (!scratch) return 0;\n",
    "        const float *m = ds4_gpu_tp_mask_router_weights(\n",
    "                (const float *)weights->ptr, (const int32_t *)selected->ptr,\n",
    "                scratch, n_expert, n_total_expert);\n",
    "        if (m != (const float *)weights->ptr) {\n",
    "            masked_weights = *weights;\n",
    "            masked_weights.ptr = (void *)m;\n",
    "            masked_weights.owner = 0;\n",
    "            use_weights = &masked_weights;\n",
    "        }\n",
    "    }\n",
    "    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,\n",
    "                             gate_offset, up_offset, down_offset,\n",
    "                             gate_type, down_type,\n",
    "                             gate_expert_bytes, gate_row_bytes,\n",
    "                             down_expert_bytes, down_row_bytes,\n",
    "                             expert_in_dim, expert_mid_dim, out_dim,\n",
    "                             selected, use_weights, n_total_expert, n_expert, clamp, x, layer_index, 1,\n",
    "                             force_resident);\n}",
])

# The forward declarations must precede the function; put them at file scope.
P9_DECL_ANCHOR = "extern \"C\" int ds4_gpu_routed_moe_one_tensor("
P9_DECL = "".join([
    "/* DS4-TP-gfx1151 (patch 9): defined in the TP runtime, which ds4_rocm.cu\n",
    " * includes AFTER this header. */\n",
    "extern \"C\" int ds4_gpu_tp_expert_shard_active(void);\n",
    "extern \"C\" const float *ds4_gpu_tp_mask_router_weights(\n",
    "        const float *weights, const int32_t *selected, float *scratch,\n",
    "        uint32_t n_pairs, uint32_t n_total_expert);\n",
    "extern \"C\" int ds4_gpu_routed_moe_one_tensor(",
])


def patch_moe_expert_mask(tree, check):
    name = "apply expert-shard weight mask in routed_moe_one"
    path = os.path.join(tree, "rocm", "ds4_rocm_moe_launch.cuh")
    if not os.path.exists(path):
        _log(name, FAIL, "missing " + path)
        return
    src = _read(path)
    if MARKER + " (patch 9)" in src:
        _log(name, ALREADY, "mask already applied")
        return
    na, nd = src.count(P9_ANCHOR), src.count(P9_DECL_ANCHOR)
    if na != 1 or nd != 1:
        _log(name, FAIL, "anchor counts body=%d decl=%d (want 1,1) - refusing" % (na, nd))
        return
    out = src.replace(P9_DECL_ANCHOR, P9_DECL).replace(P9_ANCHOR, P9_NEW)
    if _write_checked(path, src, out, check):
        _log(name, WOULD if check else APPLIED,
             "routed_moe_one masks non-owned experts under TP=2")


PATCHES = [
    patch_verbs_platform,
    patch_verbs_dlopen,
    patch_tp_backend_gate,
    patch_tp_engine_platform,
    patch_rocm_tp_runtime,
    patch_makefile_tp_flag,
    patch_qp_type_fallback,
    patch_moe_expert_mask,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/home/wkljohn/Desktop/cc/ds4-upstream",
                    help="path to a stock ds4 checkout")
    ap.add_argument("--check", action="store_true",
                    help="dry run: report what would change, write nothing")
    a = ap.parse_args()

    if not os.path.isdir(a.tree):
        print(f"FATAL: no such tree: {a.tree}")
        return 2
    if not os.path.exists(os.path.join(a.tree, "ds4_tp.c")):
        print(f"FATAL: {a.tree} does not look like a ds4 checkout (no ds4_tp.c)")
        return 2

    print(f"{'checking' if a.check else 'patching'} {a.tree}\n")
    for fn in PATCHES:
        try:
            fn(a.tree, a.check)
        except Exception as e:  # noqa: BLE001 - a patcher must never half-apply
            _log(fn.__name__, FAIL, f"{type(e).__name__}: {e}")

    n_fail = sum(1 for _, s in _RESULTS if s == FAIL)
    n_would = sum(1 for _, s in _RESULTS if s == WOULD)
    n_done = sum(1 for _, s in _RESULTS if s in (APPLIED, ALREADY))
    print()
    if n_fail:
        print(f"{n_fail} patch(es) FAILED - tree is NOT safe to build")
        return 1
    if a.check:
        print(f"{n_would} pending, {n_done} already applied")
    else:
        print(f"ok - {n_done}/{len(PATCHES)} in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
