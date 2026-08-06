# Following canonical DS4 updates

This fork has two remotes with distinct jobs:

```text
origin    git@github.com:wkljohn/ds4-strix-halo-tp-odinlink.git
upstream  https://github.com/antirez/ds4.git
```

The history records canonical DS4 commit `54b36ed` as the engine base used by
the fork. This is a real merge base: do not use `--allow-unrelated-histories`
for later updates and do not re-import the upstream tree as a bulk copy.

## Prepare an update

Start from a clean, current `main` branch:

```sh
git switch main
git pull --ff-only origin main
./scripts/prepare-upstream-sync.sh
```

The helper adds the canonical `upstream` remote when needed, fetches it, creates
a branch named `sync/upstream-<commit>`, and performs a non-committing merge of
`upstream/main`. It never commits, pushes, or silently accepts conflicts. To
sync a reviewed tag or commit instead of the tip of `upstream/main`, pass it as
the first argument:

```sh
./scripts/prepare-upstream-sync.sh upstream/main
./scripts/prepare-upstream-sync.sh v1.2.3
./scripts/prepare-upstream-sync.sh <canonical-commit>
```

## Conflict policy

Adopt canonical fixes by default, but explicitly preserve or reapply the fork's
Strix Halo TP behavior in these hotspots:

- `ds4_tp.c` and `ds4_tp.h`: TP wire protocol, provider isolation, negotiated
  OdinLink decode size, and generic/Mellanox fallback behavior.
- `ds4_rocm.cu` and `rocm/`: gfx1151 Q4_K integer-WMMA, registered-slab big
  gates, attention range work, and every runtime kill switch.
- `ds4.c`: TP=2 row ownership, independent attention/FFN split policy, and
  asynchronous gate lifetime rules.
- `Makefile`: the `strix-halo` build and its gfx1151 defaults.
- `README.md` and `ODINLINK.md`: fork setup, performance evidence, and the
  requirement that the two node filesystems be synchronized explicitly.

Never resolve protocol conflicts independently on the two machines. Produce
one candidate commit, then install that exact commit on both nodes.

## Validate before accepting the merge

At minimum, inspect the staged merge and run the local protocol/build gates:

```sh
git diff --cached --check
git diff --cached --stat
make test-tp-hello
make -j"$(nproc)" strix-halo
```

Then run the documented two-node deterministic comparison from `ODINLINK.md`.
The candidate must use identical DS4 binary, provider, model, and environment
on both independent filesystems. Check hashes on both nodes, verify generated
output, and reject unexplained prefill or decode regressions.

When the merge is accepted:

```sh
git commit -m "Merge canonical DS4 updates through <commit>"
git switch main
git merge --ff-only sync/upstream-<commit>
git push origin main
```

On each inference node, update from the same published commit and rebuild:

```sh
git fetch origin
git switch main
git pull --ff-only origin main
make -j"$(nproc)" strix-halo
git rev-parse HEAD
sha256sum ./ds4
```

The two `git rev-parse` values and binary hashes must match before a TP=2 run.

