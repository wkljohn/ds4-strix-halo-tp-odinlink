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
# NOTE deliberately using the SYSTEM libibverbs, not OdinLink's shim
# (libodl_tb5_verbs.so). The shim exports 42 of 187 symbols and is missing
# ibv_get_device_list and ibv_query_gid - both of which tp_rdma_probe() calls.
# The correct integration is the rdma-core PROVIDER
# (libodl_tb5-rdmav34.so, registered in /etc/libibverbs.d/), which lets the
# real libibverbs handle enumeration and GID lookup while dispatching device
# operations into OdinLink.
# ---------------------------------------------------------------------------
P2_ANCHOR = """    void *h = dlopen("/usr/lib/librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);"""

P2_NEW = """    /* DS4-TP-gfx1151 (patch 2): prefer the Linux verbs library. Use the
     * SYSTEM libibverbs, never OdinLink's partial shim - the shim lacks
     * ibv_get_device_list and ibv_query_gid, which tp_rdma_probe() needs.
     * OdinLink plugs in underneath as an rdma-core provider instead. */
    void *h = NULL;
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


PATCHES = [
    patch_verbs_platform,
    patch_verbs_dlopen,
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
