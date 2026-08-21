CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
SAMPLING_TEST :=
else
NATIVE_CPU_FLAG ?= -march=native
SAMPLING_TEST := tests/test_sampling
endif

DEBUG_FLAGS ?= -g
PROFILE ?= 0
ifeq ($(PROFILE),1)
DS4_PROFILE_CFLAGS := -DDS4_ENABLE_PROFILING=1
else
DS4_PROFILE_CFLAGS :=
endif
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
CFLAGS += $(DS4_PROFILE_CFLAGS)
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
ROCM_SRCS := $(wildcard rocm/*.cuh)
DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support.gguf

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CORE_OBJS = ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1151
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_CFLAGS += $(DS4_PROFILE_CFLAGS)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)
METAL_LDLIBS := $(LDLIBS)
endif

.PHONY: all help clean test test-quality-gates test-moe-wave-plan test-rocm-moe-wave-plan test-tp-hello test-roce-v2-mr test-tp-completion-ordering test-tp-big-gate-overlap test-tp-verify-gate-latency test-rocm-tp-split-gate test-rocm-prefill-wavefront-projections test-rocm-long-context test-metal-session-batch test-cuda-session-batch test-cuda-mixed-batch test-rocm-attention-output-tp test-rocm-attention-prefill-static-flash test-rocm-q4k-skip-unowned test-rocm-q4k-fused-mid test-rocm-q4k-one-token-oracle test-rocm-q4k-staged-midq-oracle test-rocm-q4k-ffn-row-balance-oracle test-rocm-q4k-slot-balance-oracle test-rocm-q4k-verify-batch-oracle test-rocm-q8-attn-out-weight-outer test-rocm-indexer-exact-token-loop test-rocm-attention-exact-head2 test-rocm-argmax-rows test-rocm-compressor-row-shard-oracle dspark-acceptance dspark-verify-depth mtp-verify-depth cpu cuda cuda-spark cuda-generic cuda-regression strix-halo strix-halo-quality-score rocm

test-quality-gates:
	python3 tests/test_frontier_logits_gate.py

test-moe-wave-plan:
	python3 tests/test_moe_wave_plan.py

tests/test_rocm_moe_wave_plan: tests/test_rocm_moe_wave_plan.cu
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $<

test-rocm-moe-wave-plan: tests/test_rocm_moe_wave_plan
	./tests/test_rocm_moe_wave_plan

tests/ds4_tp_hello_test.o: ds4_tp.c ds4_tp.h ds4.h
	$(CC) $(CFLAGS) -DDS4_TP_TEST_HOOKS -ffunction-sections -fdata-sections -c -o $@ ds4_tp.c

tests/test_tp_hello: tests/test_tp_hello.c tests/ds4_tp_hello_test.o ds4_tp.h ds4.h
	$(CC) $(CFLAGS) -DDS4_TP_TEST_HOOKS -ffunction-sections -Wl,--gc-sections -I. -o $@ tests/test_tp_hello.c tests/ds4_tp_hello_test.o $(LDLIBS)

test-tp-hello: tests/test_tp_hello
	./tests/test_tp_hello

tests/roce_v2_mr_probe: tests/roce_v2_mr_probe.cpp
	$(HIPCC) -O2 -o $@ $< -libverbs

test-roce-v2-mr: tests/roce_v2_mr_probe
	@test -n "$(RDMA_DEVICE)" || { echo "error: set RDMA_DEVICE=mlx5_N" >&2; exit 2; }
	./tests/roce_v2_mr_probe "$(RDMA_DEVICE)"

tests/ds4_tp_completion_ordering.o: ds4_tp.c ds4_tp.h ds4.h
	$(CC) $(CFLAGS) -DDS4_ROCM_BUILD -DDS4_ROCM_TP_READY=1 \
		-ffunction-sections -fdata-sections -c -o $@ ds4_tp.c

tests/test_tp_completion_ordering.o: tests/test_tp_completion_ordering.cu ds4_tp.h ds4.h
	$(HIPCC) $(ROCM_CFLAGS) -I. -c -o $@ $<

tests/test_tp_completion_ordering: tests/test_tp_completion_ordering.o tests/ds4_tp_completion_ordering.o
	$(HIPCC) $(ROCM_CFLAGS) -Wl,--gc-sections -o $@ $^ -lm -pthread -ldl

test-tp-completion-ordering: tests/test_tp_completion_ordering
	@test -n "$(ROLE)" -a -n "$(ADDRESS)" -a -n "$(RDMA_DEVICE)" || { \
		echo "error: set ROLE, ADDRESS, and RDMA_DEVICE" >&2; exit 2; }
	./tests/test_tp_completion_ordering "$(ROLE)" "$(ADDRESS)" \
		"$${PORT:-5598}" "$(RDMA_DEVICE)" "$${GID_INDEX:--1}" "$${ITERATIONS:-25800}"

tests/ds4_tp_big_gate_overlap.o: ds4_tp.c ds4_tp.h ds4.h
	$(CC) $(CFLAGS) -DDS4_ROCM_BUILD -DDS4_ROCM_TP_READY=1 \
		-ffunction-sections -fdata-sections -c -o $@ ds4_tp.c

tests/test_tp_big_gate_overlap.o: tests/test_tp_big_gate_overlap.cu ds4_tp.h ds4.h
	$(HIPCC) $(ROCM_CFLAGS) -I. -c -o $@ $<

tests/test_tp_big_gate_overlap: tests/test_tp_big_gate_overlap.o tests/ds4_tp_big_gate_overlap.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -Wl,--gc-sections -o $@ $^ $(ROCM_LDLIBS) -ldl

test-tp-big-gate-overlap: tests/test_tp_big_gate_overlap
	@test -n "$(TP_ROLE)" -a -n "$(TP_LEADER)" -a -n "$(TP_PORT)" -a -n "$(RDMA_DEVICE)" -a -n "$(RDMA_GID_INDEX)" || { \
		echo "error: set TP_ROLE TP_LEADER TP_PORT RDMA_DEVICE RDMA_GID_INDEX" >&2; exit 2; }
	./tests/test_tp_big_gate_overlap "$(TP_ROLE)" "$(TP_LEADER)" "$(TP_PORT)" \
		"$(RDMA_DEVICE)" "$(RDMA_GID_INDEX)" \
		"$(if $(TP_CHUNKS),$(TP_CHUNKS),8)" \
		"$(if $(TP_WORK_ITERS),$(TP_WORK_ITERS),4096)"

tests/test_tp_verify_gate_latency.o: tests/test_tp_verify_gate_latency.cu ds4_tp.h ds4.h
	$(HIPCC) $(ROCM_CFLAGS) -I. -c -o $@ $<

tests/test_tp_verify_gate_latency: tests/test_tp_verify_gate_latency.o tests/ds4_tp_big_gate_overlap.o
	$(HIPCC) $(ROCM_CFLAGS) -Wl,--gc-sections -o $@ $^ -lm -pthread -ldl

test-tp-verify-gate-latency: tests/test_tp_verify_gate_latency
	@test -n "$(TP_ROLE)" -a -n "$(TP_LEADER)" -a -n "$(TP_PORT)" -a -n "$(RDMA_DEVICE)" -a -n "$(RDMA_GID_INDEX)" || { \
		echo "error: set TP_ROLE TP_LEADER TP_PORT RDMA_DEVICE RDMA_GID_INDEX" >&2; exit 2; }
	./tests/test_tp_verify_gate_latency "$(TP_ROLE)" "$(TP_LEADER)" "$(TP_PORT)" \
		"$(RDMA_DEVICE)" "$(RDMA_GID_INDEX)" \
		"$(if $(TP_ITERATIONS),$(TP_ITERATIONS),200)"

tests/test_rocm_tp_split_gate.o: tests/test_rocm_tp_split_gate.cu ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_tp_split_gate: tests/test_rocm_tp_split_gate.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-tp-split-gate: tests/test_rocm_tp_split_gate
	./tests/test_rocm_tp_split_gate

tests/test_rocm_prefill_wavefront_projections.o: tests/test_rocm_prefill_wavefront_projections.cu ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_prefill_wavefront_projections: tests/test_rocm_prefill_wavefront_projections.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-prefill-wavefront-projections: tests/test_rocm_prefill_wavefront_projections
	./tests/test_rocm_prefill_wavefront_projections

tests/rocm_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h ds4_tp.h
	$(CC) $(CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/rocm_long_context_smoke: tests/rocm_long_context_smoke.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-long-context: tests/rocm_long_context_smoke
	./tests/rocm_long_context_smoke

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test         Build and run tests"
	@echo "  make dspark-verify-depth  Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth  Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS) $(METAL_LDLIBS)

gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c ds4.h ds4_distributed.h ds4_tp.h
	$(CC) $(QUALITY_CFLAGS) -I. -c -o $@ $<

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o
	$(CC) $(QUALITY_CFLAGS) -o $@ $^ $(METAL_LDLIBS)

tests/test_metal_session_batch.o: tests/test_metal_session_batch.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_metal_session_batch.c

tests/test_metal_session_batch: tests/test_metal_session_batch.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

test-metal-session-batch: tests/test_metal_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_metal_session_batch

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "  make strix-halo-quality-score  Build the ROCm official-fixture scorer"
	@echo "  make ds4-bench-tp        Build the fixed-frontier two-node TP benchmark"
	@echo "  make rocm                Alias for make strix-halo"
	@echo "  make rocm PROFILE=1      Build diagnostic profilers (disabled in production)"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run tests"
	@echo "  make dspark-verify-depth Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth    Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

strix-halo:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-bench-tp ds4-eval ds4-agent \
		CORE_OBJS="ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o" \
		CFLAGS="$(CFLAGS) -DDS4_ROCM_BUILD -DDS4_ROCM_TP_READY=1" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

strix-halo-quality-score:
	$(MAKE) -B gguf-tools/quality-testing/score_official \
		CORE_OBJS="ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_layer_pack.o" \
		CFLAGS="$(CFLAGS) -DDS4_ROCM_BUILD -DDS4_ROCM_TP_READY=1" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

rocm: strix-halo

ds4: ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

# Dedicated name for the two-node fixed-frontier harness.  It deliberately
# shares ds4-bench's measurement loop, while its launcher enforces TP + RDMA.
ds4-bench-tp: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c ds4.h ds4_distributed.h ds4_tp.h
	$(CC) $(QUALITY_CFLAGS) -I. -c -o $@ $<

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h ds4_tp.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

ds4_rocm.o: ds4_rocm.cu ds4_gpu.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS)
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

ds4_rocm_compat.o: ds4_rocm_compat.cu ds4_gpu.h ds4_gpu_mgpu.h ds4_gpu_args.h
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_compat.cu

ds4_rocm_unavailable.o: ds4_rocm_unavailable.cu
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm_unavailable.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_layer_pack.o: tests/test_layer_pack.c ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_args.o: tests/test_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c ds4.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o ds4_cpu_test_hooks.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ifneq ($(UNAME_S),Darwin)
tests/test_gpu_xdev.o: tests/test_gpu_xdev.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_xdev: tests/test_gpu_xdev.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_rocm_attention_output_tp.o: tests/test_rocm_attention_output_tp.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_attention_output_tp: tests/test_rocm_attention_output_tp.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-attention-output-tp: tests/test_rocm_attention_output_tp
	./tests/test_rocm_attention_output_tp

tests/test_rocm_attention_decode_mixed.o: tests/test_rocm_attention_decode_mixed.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_attention_decode_mixed: tests/test_rocm_attention_decode_mixed.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-attention-decode-mixed: tests/test_rocm_attention_decode_mixed
	./tests/test_rocm_attention_decode_mixed

tests/test_rocm_attention_prefill_static_flash.o: tests/test_rocm_attention_prefill_static_flash.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_attention_prefill_static_flash: tests/test_rocm_attention_prefill_static_flash.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-attention-prefill-static-flash: tests/test_rocm_attention_prefill_static_flash
	./tests/test_rocm_attention_prefill_static_flash

tests/test_rocm_q4k_skip_unowned.o: tests/test_rocm_q4k_skip_unowned.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_q4k_skip_unowned: tests/test_rocm_q4k_skip_unowned.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-q4k-skip-unowned: tests/test_rocm_q4k_skip_unowned
	DS4_ROCM_Q4K_DECODE_STAGE_XQ=1 ./tests/test_rocm_q4k_skip_unowned
	DS4_ROCM_Q4K_DECODE_STAGE_XQ=0 ./tests/test_rocm_q4k_skip_unowned

tests/test_rocm_q4k_fused_mid.o: tests/test_rocm_q4k_fused_mid.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_q4k_fused_mid: tests/test_rocm_q4k_fused_mid.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-q4k-fused-mid: tests/test_rocm_q4k_fused_mid
	@set -e; control=$$(mktemp); candidate=$$(mktemp); candidate_log=$$(mktemp); \
	 trap 'rm -f "$$control" "$$candidate" "$$candidate_log"' EXIT; \
	 DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP=1 DS4_ROCM_Q4K_WMMA_FUSE_MID=0 ./tests/test_rocm_q4k_fused_mid "$$control"; \
	 DS4_ROCM_Q4K_WMMA_PAIR_GATE_UP=1 DS4_ROCM_Q4K_WMMA_FUSE_MID=1 ./tests/test_rocm_q4k_fused_mid "$$candidate" >"$$candidate_log" 2>&1; \
	 cat "$$candidate_log"; \
	 grep -q 'Q4_K WMMA fused-mid active' "$$candidate_log"; \
	 cmp "$$control" "$$candidate"

tests/test_rocm_q4k_decode_bench.o: tests/test_rocm_q4k_decode_bench.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_q4k_decode_bench: tests/test_rocm_q4k_decode_bench.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-q4k-decode-bench: tests/test_rocm_q4k_decode_bench
	DS4_ROCM_Q4K_DECODE_STAGE_XQ=1 ./tests/test_rocm_q4k_decode_bench
	DS4_ROCM_Q4K_DECODE_STAGE_XQ=0 ./tests/test_rocm_q4k_decode_bench
	DS4_ROCM_Q4K_DECODE_STAGE_XQ=1 DS4_ROCM_Q4K_DECODE_SPLIT_GATE_UP=1 ./tests/test_rocm_q4k_decode_bench

tests/test_rocm_q4k_one_token_oracle.o: tests/test_rocm_q4k_one_token_oracle.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_q4k_one_token_oracle: tests/test_rocm_q4k_one_token_oracle.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-q4k-one-token-oracle: tests/test_rocm_q4k_one_token_oracle
	./tests/test_rocm_q4k_one_token_oracle

tests/test_rocm_q4k_staged_midq_oracle.o: tests/test_rocm_q4k_staged_midq_oracle.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_q4k_staged_midq_oracle: tests/test_rocm_q4k_staged_midq_oracle.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

tests/test_rocm_q4k_ffn_row_balance_oracle.o: tests/test_rocm_q4k_ffn_row_balance_oracle.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_q4k_ffn_row_balance_oracle: tests/test_rocm_q4k_ffn_row_balance_oracle.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-q4k-ffn-row-balance-oracle: tests/test_rocm_q4k_ffn_row_balance_oracle
	./tests/test_rocm_q4k_ffn_row_balance_oracle

tests/test_rocm_q4k_slot_balance_oracle.o: tests/test_rocm_q4k_slot_balance_oracle.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_q4k_slot_balance_oracle: tests/test_rocm_q4k_slot_balance_oracle.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-q4k-slot-balance-oracle: tests/test_rocm_q4k_slot_balance_oracle
	./tests/test_rocm_q4k_slot_balance_oracle

tests/test_rocm_q4k_verify_batch_oracle.o: tests/test_rocm_q4k_verify_batch_oracle.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_q4k_verify_batch_oracle: tests/test_rocm_q4k_verify_batch_oracle.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-q4k-verify-batch-oracle: tests/test_rocm_q4k_verify_batch_oracle
	./tests/test_rocm_q4k_verify_batch_oracle 2 8 first-owner
	DS4_TEST_SWAP_GATE_UP=1 ./tests/test_rocm_q4k_verify_batch_oracle 2 8 first-owner
	./tests/test_rocm_q4k_verify_batch_oracle 3 8 first-owner
	./tests/test_rocm_q4k_verify_batch_oracle 4 8 first-owner
	./tests/test_rocm_q4k_verify_batch_oracle 5 8 first-owner

tests/test_rocm_q8_attn_out_weight_outer.o: tests/test_rocm_q8_attn_out_weight_outer.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_q8_attn_out_weight_outer: tests/test_rocm_q8_attn_out_weight_outer.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-q8-attn-out-weight-outer: tests/test_rocm_q8_attn_out_weight_outer
	./tests/test_rocm_q8_attn_out_weight_outer 2
	./tests/test_rocm_q8_attn_out_weight_outer 3
	./tests/test_rocm_q8_attn_out_weight_outer 4
	./tests/test_rocm_q8_attn_out_weight_outer 5
	DS4_ROCM_ATTN_OUT_LOW_THREADS=256 DS4_ROCM_ATTN_OUT_EXPAND_THREADS=256 ./tests/test_rocm_q8_attn_out_weight_outer 2
	DS4_ROCM_ATTN_OUT_LOW_THREADS=256 DS4_ROCM_ATTN_OUT_EXPAND_THREADS=256 ./tests/test_rocm_q8_attn_out_weight_outer 3
	DS4_ROCM_ATTN_OUT_LOW_THREADS=256 DS4_ROCM_ATTN_OUT_EXPAND_THREADS=256 ./tests/test_rocm_q8_attn_out_weight_outer 4
	DS4_ROCM_ATTN_OUT_LOW_THREADS=256 DS4_ROCM_ATTN_OUT_EXPAND_THREADS=256 ./tests/test_rocm_q8_attn_out_weight_outer 5

tests/test_rocm_indexer_exact_token_loop.o: tests/test_rocm_indexer_exact_token_loop.cu ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_indexer_exact_token_loop: tests/test_rocm_indexer_exact_token_loop.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-indexer-exact-token-loop: tests/test_rocm_indexer_exact_token_loop
	./tests/test_rocm_indexer_exact_token_loop 1
	./tests/test_rocm_indexer_exact_token_loop 2
	./tests/test_rocm_indexer_exact_token_loop 3
	./tests/test_rocm_indexer_exact_token_loop 4
	./tests/test_rocm_indexer_exact_token_loop 5

tests/test_rocm_attention_exact_head2.o: tests/test_rocm_attention_exact_head2.cu ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_attention_exact_head2: tests/test_rocm_attention_exact_head2.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-attention-exact-head2: tests/test_rocm_attention_exact_head2
	./tests/test_rocm_attention_exact_head2 2
	./tests/test_rocm_attention_exact_head2 3
	./tests/test_rocm_attention_exact_head2 4
	./tests/test_rocm_attention_exact_head2 5

tests/test_rocm_argmax_rows.o: tests/test_rocm_argmax_rows.cu ds4_gpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_argmax_rows: tests/test_rocm_argmax_rows.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-argmax-rows: tests/test_rocm_argmax_rows
	./tests/test_rocm_argmax_rows

test-rocm-q4k-staged-midq-oracle: tests/test_rocm_q4k_staged_midq_oracle
	@set -e; control=$$(mktemp); candidate=$$(mktemp); \
	 control_log=$$(mktemp); candidate_log=$$(mktemp); \
	 trap 'rm -f "$$control" "$$candidate" "$$control_log" "$$candidate_log"' EXIT; \
	 DS4_ROCM_Q4K_DECODE_STAGE_MIDQ=0 DS4_TEST_OUTPUT_FILE="$$control" \
	   ./tests/test_rocm_q4k_staged_midq_oracle >"$$control_log" 2>&1; \
	 cat "$$control_log"; \
	 DS4_ROCM_Q4K_DECODE_STAGE_MIDQ=1 DS4_TEST_OUTPUT_FILE="$$candidate" \
	   ./tests/test_rocm_q4k_staged_midq_oracle >"$$candidate_log" 2>&1; \
	 cat "$$candidate_log"; \
	 grep -q 'Q4_K decode staged-MIDQ active' "$$candidate_log"; \
	 grep -q 'Q4_K decode staged-MIDQ active' "$$control_log" && exit 1 || true; \
	 cmp "$$control" "$$candidate"; \
	 echo "staged-midq bitwise: PASS"; \
	 python3 -c "import re,sys; \
c=open(sys.argv[1]).read(); k=open(sys.argv[2]).read(); \
mc=re.search(r'avg_ms=([0-9.]+)', c); mk=re.search(r'avg_ms=([0-9.]+)', k); \
tc=float(mc.group(1)); tk=float(mk.group(1)); \
gain=(tc-tk)/tc if tc>0 else 0.0; \
print('staged-midq shipped_avg_ms=%.6f candidate_avg_ms=%.6f gain=%.2f%% enable_gate=%s' % \
(tc, tk, 100.0*gain, 'PASS' if gain>=0.10 else 'HOLD (need >=10% vs full one-token MoE)'))" \
	   "$$control_log" "$$candidate_log"

tests/test_rocm_compressor_row_shard_oracle.o: tests/test_rocm_compressor_row_shard_oracle.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_compressor_row_shard_oracle: tests/test_rocm_compressor_row_shard_oracle.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-compressor-row-shard-oracle: tests/test_rocm_compressor_row_shard_oracle
	./tests/test_rocm_compressor_row_shard_oracle

tests/test_rocm_shared_gu_swiglu_fused.o: tests/test_rocm_shared_gu_swiglu_fused.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_shared_gu_swiglu_fused: tests/test_rocm_shared_gu_swiglu_fused.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-shared-gu-swiglu-fused: tests/test_rocm_shared_gu_swiglu_fused
	./tests/test_rocm_shared_gu_swiglu_fused

tests/test_rocm_q8_pair_pack4.o: tests/test_rocm_q8_pair_pack4.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(HIPFLAGS) -I. -c $< -o $@

tests/test_rocm_q8_pair_pack4: tests/test_rocm_q8_pair_pack4.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(HIPFLAGS) $^ -o $@ $(ROCM_LDLIBS)

test-rocm-q8-pair-pack4: tests/test_rocm_q8_pair_pack4
	./tests/test_rocm_q8_pair_pack4

tests/test_rocm_attention_q_b_fused.o: tests/test_rocm_attention_q_b_fused.cu ds4_gpu.h ds4_gpu_mgpu.h
	$(HIPCC) $(ROCM_CFLAGS) -DDS4_ROCM_BUILD -I. -c -o $@ $<

tests/test_rocm_attention_q_b_fused: tests/test_rocm_attention_q_b_fused.o ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

test-rocm-attention-q-b-fused: tests/test_rocm_attention_q_b_fused
	./tests/test_rocm_attention_q_b_fused

tests/test_gpu_model_cache.o: tests/test_gpu_model_cache.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_cache: tests/test_gpu_model_cache.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_lookup_cache_strict.o: tests/test_gpu_lookup_cache_strict.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_lookup_cache_strict: tests/test_gpu_lookup_cache_strict.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_cuda_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ ds4.c

tests/test_engine_mgpu_refusal.o: tests/test_engine_mgpu_refusal.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_refusal: tests/test_engine_mgpu_refusal.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_mgpu_runtime.o: tests/test_engine_mgpu_runtime.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_runtime: tests/test_engine_mgpu_runtime.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_correctness.o: tests/test_engine_correctness.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_correctness: tests/test_engine_correctness.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_sampling.o: tests/test_sampling.c ds4.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_session_batch.o: tests/test_cuda_session_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_session_batch: tests/test_cuda_session_batch.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-session-batch: tests/test_cuda_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_session_batch

tests/test_cuda_mixed_batch.o: tests/test_cuda_mixed_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_mixed_batch: tests/test_cuda_mixed_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_layer_pack.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-mixed-batch: tests/test_cuda_mixed_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_mixed_batch
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(DS4_LINK) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(DS4_LINK_LIBS)
endif

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test tests/test_tp_hello \
	tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args \
	$(SAMPLING_TEST) ds4 ds4-server ds4-bench ds4-agent
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_tp_hello
	./tests/test_gpu_args_cli.sh
ifneq ($(UNAME_S),Darwin)
	./tests/test_sampling
endif

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run make dspark-support or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: run ./download_model.sh mtp or set DS4_TEST_MTP=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-bench-tp ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test gguf-tools/quality-testing/score_official gguf-tools/quality-testing/score_official.o tests/test_q4k_dot tests/test_tp_hello tests/test_tp_completion_ordering tests/test_metal_session_batch tests/test_gpu_xdev tests/test_rocm_attention_output_tp tests/test_rocm_attention_decode_mixed tests/test_rocm_attention_prefill_static_flash tests/test_rocm_q4k_decode_bench tests/test_rocm_q4k_fused_mid tests/test_rocm_q4k_staged_midq_oracle tests/test_rocm_q4k_ffn_row_balance_oracle tests/test_rocm_q4k_verify_batch_oracle tests/test_rocm_q8_attn_out_weight_outer tests/test_rocm_argmax_rows tests/test_rocm_shared_gu_swiglu_fused tests/test_rocm_q8_pair_pack4 tests/test_rocm_attention_q_b_fused tests/test_gpu_model_cache tests/test_gpu_lookup_cache_strict tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness tests/test_sampling tests/test_cuda_session_batch tests/test_cuda_mixed_batch tests/*.o *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o tests/rocm_long_context_smoke tests/rocm_long_context_smoke.o
