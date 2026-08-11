#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_gpu_args.h"
#include "ds4_help.h"
#include "ds4_tp.h"

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory when the payload is small enough,
 * performs a fixed greedy decode run without allowing EOS, restores the
 * snapshot or replays the prefix, and continues to the next frontier.  Snapshot
 * save/restore time is intentionally outside both timing windows.
 */

#include <errno.h>
#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES (UINT64_C(1) << 30)

typedef struct {
    const char *model_path;
    const char *mtp_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    const char *expert_profile_path;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    double step_mul;
    const char *dump_frontier_logits_dir;
    ds4_dist_options dist;
    ds4_tp_options tp;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool cuda_tensor_parallel;
    bool dspark;
    bool dspark_strict;
    bool show_output;
    bool semantic_smoke;
    bool decode_self_check;
    bool teacher_force_control;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

/* rocprofv3 --selected-regions keeps profiling overhead out of model loading
 * and prefill. Resolve ROCTx dynamically so normal benchmark binaries do not
 * acquire a profiler-library dependency. */
static void bench_rocprof_selected_region(bool resume) {
    if (getenv("DS4_BENCH_ROCPROF_SELECTED_REGIONS") == NULL) return;
    typedef int (*control_fn)(uint64_t);
    static int initialized;
    static control_fn pause_fn;
    static control_fn resume_fn;
    static void *roctx_handle;
    if (!initialized) {
        initialized = 1;
        roctx_handle = dlopen("librocprofiler-sdk-roctx.so",
                             RTLD_NOW | RTLD_GLOBAL);
        void *scope = roctx_handle ? roctx_handle : RTLD_DEFAULT;
        pause_fn = (control_fn)dlsym(scope, "roctxProfilerPause");
        resume_fn = (control_fn)dlsym(scope, "roctxProfilerResume");
        if (!pause_fn || !resume_fn) {
            fprintf(stderr,
                    "ds4-bench: rocprof selected-region control unavailable\n");
        }
    }
    control_fn fn = resume ? resume_fn : pause_fn;
    if (fn && fn(0) != 0) {
        fprintf(stderr, "ds4-bench: rocprof selected-region %s failed\n",
                resume ? "resume" : "pause");
    }
}

static uint64_t bench_token_hash_update(uint64_t hash, int token) {
    const uint32_t value = (uint32_t)token;
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        hash ^= (uint8_t)(value >> shift);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t bench_snapshot_max_bytes(void) {
    const char *env = getenv("DS4_BENCH_SNAPSHOT_MAX_BYTES");
    if (!env || env[0] == '\0') return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    if (!strcmp(env, "unlimited") || !strcmp(env, "UNLIMITED") ||
        !strcmp(env, "inf") || !strcmp(env, "INF")) {
        return UINT64_MAX;
    }
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (env[0] == '\0' || !end || *end != '\0') {
        fprintf(stderr,
                "ds4-bench: invalid DS4_BENCH_SNAPSHOT_MAX_BYTES=%s; using default %llu\n",
                env,
                (unsigned long long)DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES);
        return DS4_BENCH_DEFAULT_SNAPSHOT_MAX_BYTES;
    }
    return (uint64_t)v;
}

static double bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_BENCH, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-bench: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        char tp_parse_err[256] = {0};
        ds4_tp_cli_parse_result tp_parse =
            ds4_tp_parse_cli_arg(arg,
                                 &i,
                                 argc,
                                 argv,
                                 &c.tp,
                                 tp_parse_err,
                                 sizeof(tp_parse_err));
        if (tp_parse == DS4_TP_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    tp_parse_err[0] ? tp_parse_err :
                        "invalid tensor-parallel option");
            exit(2);
        }
        if (tp_parse == DS4_TP_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dspark")) {
            c.dspark = true;
        } else if (!strcmp(arg, "--dspark-strict")) {
            c.dspark = true;
            c.dspark_strict = true;
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--gpu-vram")) {
            c.gpu_vram_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu-devices")) {
            c.gpu_devices_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cuda-tensor-parallel")) {
            c.cuda_tensor_parallel = true;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_full_layers = (uint32_t)v;
            c.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-bench: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-bench: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-bench: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--show-output")) {
            c.show_output = true;
        } else if (!strcmp(arg, "--semantic-smoke")) {
            c.semantic_smoke = true;
        } else if (!strcmp(arg, "--decode-self-check")) {
            c.decode_self_check = true;
        } else if (!strcmp(arg, "--teacher-force-control")) {
            c.teacher_force_control = true;
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    char tp_err[256];
    if (!ds4_tp_adopt_distributed_options(&c.tp, &c.dist,
                                          tp_err, sizeof(tp_err))) {
        fprintf(stderr, "ds4-bench: %s\n", tp_err);
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-bench: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    return c;
}

static bool semantic_answer_is_four(const char *text) {
    static const char close_tag[] = "</think>";
    const char *tail = NULL;
    const char *scan = text ? text : "";
    while ((scan = strstr(scan, close_tag)) != NULL) {
        tail = scan + sizeof(close_tag) - 1;
        scan = tail;
    }
    if (!tail) return false;
    /* Judge the final numeric answer, not every digit in the explanation.
     * Correct concise forms such as "2+2 = 4" must not fail merely because
     * the operands are repeated after </think>. */
    unsigned long last = 0;
    bool have_number = false;
    while (*tail) {
        if (*tail < '0' || *tail > '9') {
            tail++;
            continue;
        }
        unsigned long value = 0;
        do {
            const unsigned digit = (unsigned)(*tail - '0');
            if (value > (ULONG_MAX - digit) / 10u) return false;
            value = value * 10u + digit;
            tail++;
        } while (*tail >= '0' && *tail <= '9');
        last = value;
        have_number = true;
    }
    return have_number && last == 4u;
}

static int run_semantic_smoke(ds4_engine *engine, ds4_session *session) {
    static const char question[] = "What is 2+2? Answer clearly and briefly.";
    enum { max_tokens = 256 };
    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, question, DS4_THINK_HIGH, &prompt);

    char err[256] = "";
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-bench: semantic smoke prefill failed: %s\n", err);
        ds4_tokens_free(&prompt);
        return 1;
    }

    if (getenv("DS4_BENCH_EXPECT_GREEDY_TOP2") != NULL) {
        uint64_t probe_rng = UINT64_C(1);
        ds4_token_score score[2];
        const int vocab = ds4_engine_vocab_size(engine);
        float *logits = malloc((size_t)vocab * sizeof(*logits));
        const bool fail_closed = logits &&
            ds4_session_sample(session, 0.7f, 0, 1.0f, 0.0f,
                               &probe_rng) < 0 &&
            ds4_session_top_logprobs(session, score, 2) == 0 &&
            ds4_session_token_logprob(session, 0, &score[0]) == 0 &&
            ds4_session_copy_logits(session, logits, vocab) == 0;
        free(logits);
        if (!fail_closed) {
            fprintf(stderr,
                    "ds4-bench: greedy top2 incomplete-logits consumers did not fail closed\n");
            ds4_tokens_free(&prompt);
            return 1;
        }
        fprintf(stderr,
                "ds4-bench: greedy top2 incomplete-logits consumers fail closed\n");
    }

    size_t text_len = 0;
    size_t text_cap = 4096;
    char *text = malloc(text_cap);
    if (!text) {
        fprintf(stderr, "ds4-bench: semantic smoke out of memory\n");
        ds4_tokens_free(&prompt);
        return 1;
    }
    text[0] = '\0';

    uint64_t rng = UINT64_C(1);
    uint64_t hash = UINT64_C(14695981039346656037);
    int generated = 0;
    bool stopped = false;
    const bool speculative =
        ds4_engine_mtp_draft_tokens(engine) > 1 &&
        getenv("DS4_MTP_SPEC_DISABLE") == NULL;

    while (generated < max_tokens && !stopped) {
        int accepted[17];
        int accepted_n = 1;
        accepted[0] = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (accepted[0] < 0) {
            snprintf(err, sizeof(err), "sampling failed");
            accepted_n = -1;
        } else if (ds4_token_is_stop_for_think_mode(
                       engine, accepted[0], DS4_THINK_HIGH)) {
            stopped = true;
            break;
        } else if (speculative) {
            accepted_n = ds4_session_eval_speculative_argmax(
                session, accepted[0], max_tokens - generated,
                ds4_token_eos(engine), accepted,
                (int)(sizeof(accepted) / sizeof(accepted[0])),
                err, sizeof(err));
        }

        if (accepted_n < 0) {
            fprintf(stderr, "ds4-bench: semantic smoke decode failed: %s\n", err);
            free(text);
            ds4_tokens_free(&prompt);
            return 1;
        }

        for (int i = 0; i < accepted_n && generated < max_tokens; i++) {
            const int token = accepted[i];
            if (ds4_token_is_stop_for_think_mode(engine, token, DS4_THINK_HIGH)) {
                stopped = true;
                break;
            }
            size_t piece_len = 0;
            char *piece = ds4_token_text(engine, token, &piece_len);
            if (piece_len > SIZE_MAX - text_len - 1) {
                free(piece);
                free(text);
                ds4_tokens_free(&prompt);
                return 1;
            }
            if (text_len + piece_len + 1 > text_cap) {
                size_t next_cap = text_cap;
                while (next_cap < text_len + piece_len + 1) next_cap *= 2;
                char *next = realloc(text, next_cap);
                if (!next) {
                    free(piece);
                    free(text);
                    ds4_tokens_free(&prompt);
                    return 1;
                }
                text = next;
                text_cap = next_cap;
            }
            if (piece_len) memcpy(text + text_len, piece, piece_len);
            text_len += piece_len;
            text[text_len] = '\0';
            free(piece);
            hash = bench_token_hash_update(hash, token);
            generated++;
        }

        if (!speculative && !stopped && generated < max_tokens) {
            if (ds4_session_eval(session, accepted[0], err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: semantic smoke decode failed: %s\n", err);
                free(text);
                ds4_tokens_free(&prompt);
                return 1;
            }
        }
    }

    const bool valid = stopped && semantic_answer_is_four(text);
    fprintf(stderr,
            "ds4-bench: semantic smoke %s tokens=%d fnv64=%016llx output=\"%s\"\n",
            valid ? "passed" : "FAILED",
            generated,
            (unsigned long long)hash,
            text);
    free(text);
    ds4_tokens_free(&prompt);
    return valid ? 0 : 1;
}

static int run_decode_self_check(ds4_engine *engine, ds4_session *incremental,
                                 int ctx_size) {
    static const char question[] = "What is 2+2? Answer clearly and briefly.";
    static const int checkpoints[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    enum { steps = 256 };
    const int ncheck = (int)(sizeof(checkpoints) / sizeof(checkpoints[0]));
    const int vocab = ds4_engine_vocab_size(engine);
    const size_t logits_per_check = (size_t)vocab;

    ds4_tokens prompt = {0};
    ds4_encode_chat_prompt(engine, NULL, question, DS4_THINK_HIGH, &prompt);
    char err[256] = "";
    if (ds4_session_sync(incremental, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-bench: decode self-check prefill failed: %s\n", err);
        ds4_tokens_free(&prompt);
        return 1;
    }

    int *tokens = malloc((size_t)steps * sizeof(tokens[0]));
    float *incremental_logits = malloc((size_t)ncheck * logits_per_check *
                                       sizeof(incremental_logits[0]));
    float *batched_logits = malloc(logits_per_check * sizeof(batched_logits[0]));
    if (!tokens || !incremental_logits || !batched_logits) {
        fprintf(stderr, "ds4-bench: decode self-check out of memory\n");
        free(tokens);
        free(incremental_logits);
        free(batched_logits);
        ds4_tokens_free(&prompt);
        return 1;
    }

    int check_index = 0;
    for (int step = 1; step <= steps; step++) {
        tokens[step - 1] = ds4_session_argmax(incremental);
        if (tokens[step - 1] < 0 ||
            ds4_session_eval(incremental, tokens[step - 1],
                             err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4-bench: decode self-check incremental step %d failed: %s\n",
                    step, err);
            free(tokens);
            free(incremental_logits);
            free(batched_logits);
            ds4_tokens_free(&prompt);
            return 1;
        }
        if (check_index < ncheck && step == checkpoints[check_index]) {
            if (ds4_session_copy_logits(
                    incremental,
                    incremental_logits + (size_t)check_index * logits_per_check,
                    vocab) != vocab) {
                fprintf(stderr,
                        "ds4-bench: decode self-check could not copy incremental logits at %d\n",
                        step);
                free(tokens);
                free(incremental_logits);
                free(batched_logits);
                ds4_tokens_free(&prompt);
                return 1;
            }
            check_index++;
        }
    }

    int argmax_mismatches = 0;
    int first_argmax_mismatch = 0;
    for (int ci = 0; ci < ncheck; ci++) {
        ds4_tokens full = {0};
        ds4_tokens_copy(&full, &prompt);
        for (int i = 0; i < checkpoints[ci]; i++) {
            ds4_tokens_push(&full, tokens[i]);
        }

        ds4_session *batched = NULL;
        if (ds4_session_create(&batched, engine, ctx_size) != 0 ||
            ds4_session_sync(batched, &full, err, sizeof(err)) != 0 ||
            ds4_session_copy_logits(batched, batched_logits, vocab) != vocab) {
            fprintf(stderr,
                    "ds4-bench: decode self-check batched checkpoint %d failed: %s\n",
                    checkpoints[ci], err);
            ds4_session_free(batched);
            ds4_tokens_free(&full);
            free(tokens);
            free(incremental_logits);
            free(batched_logits);
            ds4_tokens_free(&prompt);
            return 1;
        }

        const float *inc = incremental_logits + (size_t)ci * logits_per_check;
        double sum_sq = 0.0;
        float max_abs = 0.0f;
        int differing = 0;
        int inc_argmax = -1;
        float inc_best = -INFINITY;
        for (int i = 0; i < vocab; i++) {
            float d = fabsf(inc[i] - batched_logits[i]);
            if (!isfinite(d)) d = INFINITY;
            if (d > max_abs) max_abs = d;
            sum_sq += (double)d * (double)d;
            if (memcmp(&inc[i], &batched_logits[i], sizeof(float)) != 0) differing++;
            if (inc_argmax < 0 || inc[i] > inc_best) {
                inc_argmax = i;
                inc_best = inc[i];
            }
        }
        const int batch_argmax = ds4_session_argmax(batched);
        if (inc_argmax != batch_argmax) {
            argmax_mismatches++;
            if (!first_argmax_mismatch) first_argmax_mismatch = checkpoints[ci];
        }
        fprintf(stderr,
                "ds4-bench: decode self-check step=%d incremental_argmax=%d "
                "batched_argmax=%d max_abs=%g rms=%g differing=%d\n",
                checkpoints[ci], inc_argmax, batch_argmax, max_abs,
                sqrt(sum_sq / (double)vocab), differing);

        ds4_session_free(batched);
        ds4_tokens_free(&full);
    }
    fprintf(stderr,
            "ds4-bench: decode self-check complete steps=%d checkpoints=%d "
            "argmax_mismatches=%d first_argmax_mismatch=%d\n",
            steps, ncheck, argmax_mismatches, first_argmax_mismatch);

    free(tokens);
    free(incremental_logits);
    free(batched_logits);
    ds4_tokens_free(&prompt);
    return 0;
}

static int run_teacher_force_control(ds4_engine *engine, ds4_session *session) {
    static const char question[] = "What is 2+2? Answer clearly and briefly.";
    static const char reference_text[] =
        "1.  **Analyze the User's Request**:\n"
        "    *   Question: \"What is 2+2?\"\n"
        "    *   Constraint: \"Answer clearly and briefly.\"\n\n"
        "2.  **Determine the Answer**:\n"
        "    *   2 + 2 = 4.\n\n"
        "3.  **Format the Output**:\n"
        "    *   Keep it extremely brief and clear. No extra fluff."
        "</think>4";

    ds4_tokens prompt = {0};
    ds4_tokens reference = {0};
    ds4_encode_chat_prompt(engine, NULL, question, DS4_THINK_HIGH, &prompt);
    ds4_tokenize_rendered_chat(engine, reference_text, &reference);
    ds4_tokens_push(&reference, ds4_token_eos(engine));

    if (getenv("DS4_BENCH_TRACE_TEACHER_TOKENS")) {
        fprintf(stderr, "ds4-bench: teacher-force prompt tokens=%d ids=", prompt.len);
        for (int i = 0; i < prompt.len; i++) {
            fprintf(stderr, "%s%d", i ? "," : "", prompt.v[i]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "ds4-bench: teacher-force reference tokens=%d ids=", reference.len);
        for (int i = 0; i < reference.len; i++) {
            fprintf(stderr, "%s%d", i ? "," : "", reference.v[i]);
        }
        fprintf(stderr, "\n");
    }

    char err[256] = "";
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4-bench: teacher-force control prefill failed: %s\n", err);
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&reference);
        return 1;
    }

    int mismatches = 0;
    int first_mismatch = 0;
    int near_tie_mismatches = 0;
    float worst_teacher_gap = 0.0f;
    for (int step = 0; step < reference.len; step++) {
        ds4_token_score top[2];
        ds4_token_score teacher;
        if (ds4_session_top_logprobs(session, top, 2) != 2 ||
            ds4_session_token_logprob(session, reference.v[step], &teacher) != 1) {
            fprintf(stderr,
                    "ds4-bench: teacher-force control could not read logits at step %d\n",
                    step);
            ds4_tokens_free(&prompt);
            ds4_tokens_free(&reference);
            return 1;
        }
        if (top[0].id != reference.v[step]) {
            const float margin = top[0].logit - top[1].logit;
            const float teacher_gap = top[0].logit - teacher.logit;
            mismatches++;
            if (!first_mismatch) first_mismatch = step + 1;
            if (margin <= 0.25f || teacher_gap <= 0.25f) near_tie_mismatches++;
            if (teacher_gap > worst_teacher_gap) worst_teacher_gap = teacher_gap;
            fprintf(stderr,
                    "ds4-bench: teacher-force mismatch step=%d teacher=%d top1=%d "
                    "top2=%d top1_margin=%g teacher_gap=%g\n",
                    step + 1, reference.v[step], top[0].id, top[1].id,
                    margin, teacher_gap);
        }
        if (step + 1 < reference.len &&
            ds4_session_eval(session, reference.v[step], err, sizeof(err)) != 0) {
            fprintf(stderr,
                    "ds4-bench: teacher-force control decode step %d failed: %s\n",
                    step + 1, err);
            ds4_tokens_free(&prompt);
            ds4_tokens_free(&reference);
            return 1;
        }
    }
    fprintf(stderr,
            "ds4-bench: teacher-force control complete tokens=%d mismatches=%d "
            "first_mismatch=%d near_tie_mismatches=%d worst_teacher_gap=%g\n",
            reference.len, mismatches, first_mismatch, near_tie_mismatches,
            worst_teacher_gap);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&reference);
    return 0;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4-bench: failed to copy frontier logits at %d\n", frontier);
        free(logits);
        return 1;
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = ds4_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            ds4_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4-bench: failed to close %s\n", path);
        free(logits);
        return 1;
    }
    free(logits);
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk,
                               bool        ssd_streaming) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill_mode(backend,
                                                      ctx_size,
                                                      prefill_chunk,
                                                      ssd_streaming);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-bench: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-bench: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-bench: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void maybe_warn_distributed_step_shape(const bench_config *cfg, ds4_session *session) {
    if (!cfg || !session || cfg->dist.role != DS4_DISTRIBUTED_COORDINATOR) return;
    uint32_t chunk = cfg->dist.prefill_chunk;
    if (chunk == 0) {
        const int cap = ds4_session_prefill_cap(session);
        if (cap > 0) chunk = (uint32_t)cap;
    }
    if (chunk == 0) return;
    if (cfg->step_mul == 1.0 &&
        cfg->step_incr > 0 &&
        (uint32_t)cfg->step_incr < chunk &&
        cfg->ctx_start < cfg->ctx_max)
    {
        fprintf(stderr,
                "ds4-bench: note: --step-incr=%d is smaller than distributed prefill chunk %u; "
                "suffix rows will not show multi-chunk pipeline overlap\n",
                cfg->step_incr,
                chunk);
    }
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    /* Hint the packer at the largest ctx this bench run will exercise
     * so per-layer KV bytes are priced for the real session size, not
     * a stale 4096 default. Single-tier and CPU paths ignore this. */
    int placement_ctx_hint = cfg.ctx_max;
    if (cfg.ctx_alloc > placement_ctx_hint) placement_ctx_hint = cfg.ctx_alloc;

    ds4_gpu_config gpu_cfg = {0};
    bool skip_cuda = false;
    const bool have_gpu_config = cfg.gpu_vram_arg || cfg.gpu_devices_arg;
    if (have_gpu_config) {
        char gpu_err[256];
        if (parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg,
                               &gpu_cfg, &skip_cuda,
                               gpu_err, sizeof(gpu_err)) != 0) {
            fprintf(stderr, "ds4-bench: %s\n", gpu_err);
            return 2;
        }
        cfg.backend = skip_cuda ? DS4_BACKEND_CPU : DS4_BACKEND_CUDA;
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .context_size = cfg.ctx_alloc,
        .prefill_chunk = cfg.prefill_chunk,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_full_layers = cfg.ssd_streaming_full_layers,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .cuda_tensor_parallel = cfg.cuda_tensor_parallel,
        .dspark = cfg.dspark,
        .dspark_strict = cfg.dspark_strict,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .ssd_streaming_full_layers_set = cfg.ssd_streaming_full_layers_set,
        .expert_profile_path = cfg.expert_profile_path,
        .distributed = cfg.dist,
        .tp = cfg.tp,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        return 2;
    }
    char tp_err[256];
    if (!ds4_tp_validate_engine_options(&opt, tp_err, sizeof(tp_err))) {
        fprintf(stderr, "ds4-bench: %s\n", tp_err);
        return 2;
    }
    ds4_engine *engine = NULL;
    if (have_gpu_config && !skip_cuda) {
        const bool was_auto =
            (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "auto")) ||
            (!cfg.gpu_vram_arg && cfg.gpu_devices_arg);
        char layout[256];
        if (format_gpu_layout_line(&gpu_cfg, was_auto,
                                   layout, sizeof(layout)) > 0) {
            fprintf(stdout, "%s\n", layout);
            fflush(stdout);
        }
        if (ds4_engine_create_with_gpu_config(
                &engine, &opt, &gpu_cfg) != 0) return 1;
    } else if (ds4_engine_open(&engine, &opt) != 0) {
        return 1;
    }
    ds4_tp *tp_leader = NULL;
    if (cfg.tp.role == DS4_TP_LEADER) {
        char tp_err[256] = "";
        ds4_tp_identity tp_id = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = (uint32_t)cfg.ctx_alloc,
            .runtime_features = ds4_engine_tp_runtime_features(engine),
        };
        ds4_engine_tp_gate_schedule(engine,
                                    &tp_id.gate_slot_start,
                                    &tp_id.gate_slot_step,
                                    &tp_id.gates_per_token);
        if (!ds4_tp_create(&tp_leader, &cfg.tp, &tp_id,
                           tp_err, sizeof(tp_err)) ||
            !ds4_engine_tp_bind(engine, tp_leader,
                                tp_err, sizeof(tp_err))) {
            fprintf(stderr, "ds4-bench-tp: %s\n", tp_err);
            ds4_tp_free(tp_leader);
            ds4_engine_close(engine);
            return 1;
        }
    }
    log_context_memory(opt.backend,
                       cfg.ctx_alloc,
                       ds4_engine_prefill_chunk(engine),
                       cfg.ssd_streaming);

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    ds4_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        ds4_encode_chat_prompt(engine, cfg.system, text, DS4_THINK_NONE, &prompt);
    } else {
        ds4_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "ds4-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        ds4_tokens_free(&prompt);
        if (tp_leader) ds4_tp_send_stop(tp_leader);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        if (tp_leader) ds4_tp_send_stop(tp_leader);
        ds4_engine_close(engine);
        return 1;
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        if (tp_leader) ds4_tp_send_stop(tp_leader);
        ds4_engine_close(engine);
        return 1;
    }
    if (cfg.decode_self_check) {
        const int self_check_rc =
            run_decode_self_check(engine, session, cfg.ctx_alloc);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        if (tp_leader) ds4_tp_send_stop(tp_leader);
        ds4_engine_close(engine);
        return self_check_rc;
    }
    if (cfg.teacher_force_control) {
        const int teacher_rc = run_teacher_force_control(engine, session);
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        if (tp_leader) ds4_tp_send_stop(tp_leader);
        ds4_engine_close(engine);
        return teacher_rc;
    }
    if (cfg.semantic_smoke) {
        if (run_semantic_smoke(engine, session) != 0) {
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            if (tp_leader) ds4_tp_send_stop(tp_leader);
            ds4_engine_close(engine);
            return 1;
        }
        /* A preflight must not share KV/cache/session state with the measured
         * workload.  Recreate the session while retaining resident weights. */
        ds4_session_free(session);
        session = NULL;
        if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
            fprintf(stderr,
                    "ds4-bench: failed to recreate session after semantic smoke\n");
            ds4_tokens_free(&prompt);
            if (tp_leader) ds4_tp_send_stop(tp_leader);
            ds4_engine_close(engine);
            return 1;
        }
    }
    maybe_warn_distributed_step_shape(&cfg, session);

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            if (tp_leader) ds4_tp_send_stop(tp_leader);
            ds4_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes,gen_cycles,gen_token_fnv64");
    for (int i = 1; i <= 17; i++) fprintf(out, ",accept_len_%d", i);
    fputc('\n', out);
    fflush(out);

    const int eos = ds4_token_eos(engine);
    const bool distributed =
        cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR ||
        cfg.tp.role == DS4_TP_LEADER;
    ds4_session_snapshot snap = {0};
    const uint64_t snapshot_max_bytes = bench_snapshot_max_bytes();
    bool warned_large_snapshot = false;
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        const double prefill_t0 = bench_now_sec();
        if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = frontier - previous;

        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        const bool need_restore_after_generation =
            cfg.gen_tokens > 0 && frontier < cfg.ctx_max;
        bool have_snapshot = false;
        if (need_restore_after_generation && !distributed &&
            getenv("DS4_BENCH_DISABLE_SNAPSHOT") == NULL) {
            const uint64_t payload_bytes = ds4_session_payload_bytes(session);
            const bool large_snapshot_forced =
                getenv("DS4_BENCH_FORCE_SNAPSHOT") != NULL;
            if (payload_bytes > snapshot_max_bytes && !large_snapshot_forced) {
                if (!warned_large_snapshot) {
                    fprintf(stderr,
                            "ds4-bench: session payload snapshot is %.2f GiB, above the %.2f GiB benchmark limit; "
                            "replaying prefixes instead (set DS4_BENCH_FORCE_SNAPSHOT=1 to force snapshots)\n",
                            bytes_to_gib(payload_bytes),
                            bytes_to_gib(snapshot_max_bytes));
                    warned_large_snapshot = true;
                }
            } else if (payload_bytes > 0) {
                if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                    fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                have_snapshot = true;
            }
        }

        bench_rocprof_selected_region(true);
        const double gen_t0 = bench_now_sec();
        double gen_first_sec = 0.0;
        double gen_steady_sec = 0.0;
        int gen_done = 0;
        int gen_first_tokens = 0;
        uint64_t gen_cycles = 0;
        uint64_t gen_token_hash = UINT64_C(14695981039346656037);
        uint64_t accept_len_hist[17] = {0};
        int *gen_token_buf = cfg.show_output && cfg.gen_tokens > 0
            ? malloc((size_t)cfg.gen_tokens * sizeof(gen_token_buf[0]))
            : NULL;
        int gen_token_count = 0;
        const bool speculative =
            ds4_engine_mtp_draft_tokens(engine) > 1 &&
            getenv("DS4_MTP_SPEC_DISABLE") == NULL;
        while (gen_done < cfg.gen_tokens) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const int token = ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const double token_t0 = bench_now_sec();
            int accepted[17];
            int accepted_n = 1;
            accepted[0] = token;
            if (speculative) {
                accepted_n = ds4_session_eval_speculative_argmax(
                    session, token, cfg.gen_tokens - gen_done, eos,
                    accepted, (int)(sizeof(accepted) / sizeof(accepted[0])),
                    err, sizeof(err));
            } else if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                accepted_n = -1;
            }
            if (accepted_n < 0) {
                fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
            if (accepted_n == 0) {
                fprintf(stderr, "ds4-bench: decode at frontier %d made no progress\n", frontier);
                rc = 1;
                break;
            }
            gen_cycles++;
            if (accepted_n <= (int)(sizeof(accept_len_hist) /
                                    sizeof(accept_len_hist[0]))) {
                accept_len_hist[accepted_n - 1]++;
            }
            const double token_t1 = bench_now_sec();
            if (gen_done == 0) {
                gen_first_sec = token_t1 - token_t0;
                gen_first_tokens = accepted_n;
            } else {
                gen_steady_sec += token_t1 - token_t0;
            }
            for (int j = 0; j < accepted_n && gen_done < cfg.gen_tokens; j++) {
                gen_token_hash = bench_token_hash_update(gen_token_hash,
                                                         accepted[j]);
                if (gen_token_buf) gen_token_buf[gen_token_count++] = accepted[j];
                gen_done++;
            }
        }
        const double gen_t1 = bench_now_sec();
        bench_rocprof_selected_region(false);
        if (cfg.show_output && gen_token_buf && gen_token_count > 0) {
            fprintf(stderr, "ds4-bench: gen[ctx=%d] decoded text: \"", frontier);
            for (int i = 0; i < gen_token_count; i++) {
                size_t tlen = 0;
                char *txt = ds4_token_text(engine, gen_token_buf[i], &tlen);
                if (txt) {
                    fwrite(txt, 1, tlen, stderr);
                    free(txt);
                }
            }
            fprintf(stderr, "\"\n");
            fflush(stderr);
        }
        free(gen_token_buf);
        if (rc != 0) break;

        if (!need_restore_after_generation) {
            /* Nothing later depends on the frontier state. */
        } else if (distributed || !have_snapshot) {
            if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: replay restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        } else {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_sec = gen_t1 - gen_t0;
        const int gen_steady_tokens =
            gen_done > gen_first_tokens ? gen_done - gen_first_tokens : 0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%.3f,%d,%.2f,%llu,%llu,%016llx",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                gen_done,
                gen_sec > 0.0 ? (double)gen_done / gen_sec : 0.0,
                gen_first_sec * 1000.0,
                gen_steady_tokens,
                gen_steady_sec > 0.0 ? (double)gen_steady_tokens / gen_steady_sec : 0.0,
                (unsigned long long)(have_snapshot ? snap.len : 0),
                (unsigned long long)gen_cycles,
                (unsigned long long)gen_token_hash);
        for (int i = 0; i < 17; i++) {
            fprintf(out, ",%llu", (unsigned long long)accept_len_hist[i]);
        }
        fputc('\n', out);
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    if (tp_leader) ds4_tp_send_stop(tp_leader);
    ds4_engine_close(engine);
    return rc;
}
