# HybridMoE

A Strata-style hybrid MoE runtime for an 8 GB AMD GPU without ROCm, built as a
fork of llama.cpp's Vulkan backend. The GPU keeps the hottest experts in VRAM,
and the CPU computes the experts a token needs that are not there, in place and
while the GPU works, instead of uploading them over PCIe.

Machine it was built and measured on: i7-12700K, 15 GB RAM (~35 GB/s), RX 580
8 GB (Polaris, RADV, PCIe 3.0 x16), NVMe SSD. Model: Ornith-1.5-27B-A3B-CoderX
Q4_K_M (qwen35moe, 41 layers, 184 experts with 8 used, 16 GB: it does not fit
in RAM, so the OS pages experts in from the SSD).

## Results

Generation, tokens/s, `llama-bench -n 128 -t 8 --repack 0 -dev Vulkan1`,
median of the last 4 of 8 repetitions (the first ones read the model from the
SSD), via `bench/bench.py`:

| Mode | upstream Vulkan | this branch |
|---|---:|---:|
| stock split, experts of 24 layers on the CPU (`-ncmoe 24`, best N) | 24.3 | 28.4 |
| stock MoE cache 5800 MiB (`--moe-cache-mib`, uploads misses) | 17.2 | not rerun |
| **hybrid**, cache 5800 MiB, no replacements (`--moe-hybrid 8 --moe-hybrid-uploads 0`) | 17.8 | **32.6** |
| hybrid, cache 5800 MiB, 1 replacement per layer step | 17.4 | 31.2 |
| dense Ornith-1.5-9B Q4_K_M fully on the GPU (unchanged path) | 31.1 | 31.6 |

The hybrid mode runs 94-95% of the selected experts on the GPU; the CPU's
share costs about 0.1 ms per layer, overlapped with the GPU.

What made the difference, measured with `GGML_SCHED_TIMING=1` and
`GGML_VK_TIMING=1` (host time per token):

1. **Vulkan submits.** ggml-vulkan submitted a graph in pieces sized by
   flops, about 7 `vkQueueSubmit` per graph here, each 45-110 us of host time.
   A model split between CPU and GPU runs 50-82 small graphs per token, so the
   GPU sat waiting on submits: 74.8 ms of host time per token. With a graph
   submitted every 100 nodes and at its end (upstream behaviour is still there
   with `GGML_VK_EARLY_SUBMITS=1`): 1.5 submits per graph, 29.6 ms.
2. **Pageable host memory.** Vulkan copies from pageable memory through a
   staging buffer and then waits for the GPU to finish everything, so each
   slot-map, CPU-result and expert upload was a full round trip. They now go
   through pinned memory and run in the GPU's queue, and the scheduler no longer
   waits for the GPU before calling the copy callback (callbacks that need it
   synchronize themselves).

## How it works

Per MoE layer, at the point where the scheduler copies the layer's slot map:

1. read the selected experts and the layer input back from the GPU (one sync);
2. experts in the VRAM cache map to their slot; the others map to an all-zero
   slot, so the GPU's `mul_mat_id` adds nothing for them;
3. the CPU pool (`--moe-hybrid N` threads) computes the missing experts with
   ggml-cpu's `vec_dot` kernels (gate/up, SiLU*up, down) into a pinned buffer,
   while the GPU computes the cached ones;
4. the CPU's result is copied in the GPU's queue and added after the experts;
5. admission: experts are counted per layer (counts halve every 64 steps); a
   free slot is filled, and with `--moe-hybrid-uploads N` up to N experts per
   step replace a colder one if used at least once more. Uploads are staged in
   pinned memory and become visible at the layer's next step.

Prompt batches over 32 tokens use the stock path (the cache is read, misses
are copied to the GPU).

## Build and run

    ./build.sh            # llama-bench, llama-cli, llama-server, llama-perplexity

    llama.cpp/build/bin/llama-server -m ~/Downloads/Ornith-1.5-27B-A3B-CoderX.Q4_K_M.gguf \
        -dev Vulkan1 -ngl 99 -ncmoe 41 --moe-cache-mib 5600 --moe-hybrid 8 -t 8 -c 8192 -fa on

`-dev Vulkan1` is the RX 580 (Vulkan0 is the iGPU). `-ncmoe 41` keeps every
expert in host memory, from where the cache takes the hot ones. Peak VRAM with
a 5800 MiB cache and a small context was 7.3 GB.

## Files

- `llama.cpp/`: the fork, branch `hybrid-moe` (git-ignored here, see `patches/`)
- `patches/`: the fork's commits on top of upstream `a0e6dc738`
- `bench/bench.py`: steady-state llama-bench runs, with disk reads; results in `bench/results.jsonl`
- `tools/gguf_header.py`: reads a GGUF header, also of a partial download
- `deps/`: Vulkan and SPIR-V headers (header-only, no root needed)

## Not done yet

- Accuracy check: KL divergence of the hybrid against the stock path with
  `llama-perplexity --kl-divergence`.
- Measured after the submit fix: other `-ncmoe` values, the stock MoE cache,
  cache size, thread count, prompt processing speed in hybrid mode.
- Speculative decoding with the model's MTP head (`--spec-type draft-mtp`):
  verifying 2-3 tokens per pass would spread the per-layer sync over them.
