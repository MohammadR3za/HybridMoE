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

The fork and two header-only Khronos packages are not part of this repository.
In a fresh clone they go next to the scripts:

    git clone -b hybrid-moe https://github.com/MohammadR3za/llama.cpp-hybrid-moe llama.cpp
    git clone -b v1.4.363 https://github.com/KhronosGroup/Vulkan-Headers deps/vulkan-headers
    git clone https://github.com/KhronosGroup/SPIRV-Headers deps/spirv-src
    cmake -S deps/spirv-src -B deps/spirv-src/build -DCMAKE_INSTALL_PREFIX="$PWD/deps/spirv"
    cmake --install deps/spirv-src/build

Then:

    ./build.sh            # llama-bench, llama-cli, llama-server, llama-perplexity

    llama.cpp/build/bin/llama-server -m ~/Downloads/Ornith-1.5-27B-A3B-CoderX.Q4_K_M.gguf \
        -dev Vulkan1 -ngl 99 -ncmoe 41 --moe-cache-mib 5600 --moe-hybrid 8 -t 8 -c 8192 -fa on

`-dev Vulkan1` is the RX 580 (Vulkan0 is the iGPU). `-ncmoe 41` keeps every
expert in host memory, from where the cache takes the hot ones. Peak VRAM with
a 5800 MiB cache and a small context was 7.3 GB.

## Qwen3.8-27B (dense)

    ./run-qwen27b.sh      # opens http://127.0.0.1:8080 once the model has loaded

The script starts llama-server with the settings below and opens the chat page
when the model is ready: 40 s from the SSD, 12-14 s while the file is still in
the page cache. This llama.cpp's server listens on port 9931 unless given
`--port`, so a command without it seemed not to start at all.

The model (Qwen3.8 27B Abliterated, Q4_K_M, 16.8 GB) is dense: 64 layers, three
Gated DeltaNet layers to each full-attention one, and one MTP layer. Its 248K
vocabulary makes the output head alone 995 MiB of Q6_K. `-ngl 28` puts the head
and blocks 38-64 on the RX 580 (7.0 GB), and the CPU reads blocks 0-37, 8.75 GB,
from RAM for every token. The RAM is a single DDR5-4800 module, one channel of
38 GB/s, so plain generation runs at about 3 tokens/s.

With the model's MTP head (`--spec-type draft-mtp --spec-draft-n-max 3`), the
GPU drafts 3 tokens (7.5 ms each) and the full model checks them in one pass,
which costs little more than one token while the CPU's share is bound by memory:
404 ms for a batch of 4 against 340-386 ms for 1 (`llama-bench -p 1,2,4,8`;
8 tokens take 844 ms, compute-bound). Plain generation is 2.94-2.98 tokens/s
whatever the text; with MTP, llama-server, 256 tokens per request:

| Request | MTP, 3 drafts | tokens per pass |
|---|---:|---:|
| code, temperature 0, no thinking | 7.79 | 3.7 |
| chat with thinking, the model's sampling (temp 1.0, top-k 20), 3 prompts | 5.20 | 2.5 |

At temperature 0 the text is identical with and without drafts.

Measured and not used:

- 2 drafts: 5.16 on chat. 4 drafts: 7.93 on code. Drafts sampled and checked by
  rejection sampling (`--spec-draft-sampling probabilistic`): the same acceptance.
- `-t 12` and `-t 16`: within 2% of `-t 8`.
- Repacked CPU weights (`--no-host`): 2.92 plain against 2.98, 7.54 with MTP
  against 7.82, and prompts at 28.5 tokens/s against 55.5, as repacked weights
  cannot be copied to the GPU for large batches.
- Pinned host memory (`--load-mode none`): the RX 580's context was lost
  (`ErrorDeviceLost`; kernel: "Not enough memory for command submission").
- The iGPU for blocks 0-37 (`-dev Vulkan0/Vulkan1 -ts 38/28`): 1.83 plain,
  prompts at 19.6.
- 29 and 30 GPU layers, with the 16K context below: 7.04 and 6.66 against 7.79,
  with 8.0 and 8.2 of the 8 GB in use.
- A Q3_K_M copy, requantized from this file to measure its speed (12.9 GiB):
  3.99 plain against 2.94, but 8.13 with MTP at 34 GPU layers against 7.79.
  Q3_K costs more to compute, and the 4-token check is compute-bound on it.

The script's other settings:

- A 16K context with the KV cache in q8_0 (target and draft) and
  `--spec-draft-n-ubatch 16`: 7750 MiB, as fast as 8K in f16 (7.79 against 7.82).
- `--ctx-checkpoints 4 --cache-ram 1024`: the server's defaults allow 32 copies
  of the recurrent state (150 MiB each here) and 8 GB of old prompts, RAM that
  the 9 GB of weights the CPU reads need.
- The fork drops the page cache of the weights a GPU holds once they are
  uploaded (patch 0004). With 15.7 GB of RAM: 0.75 GB free after loading before,
  4.2 GB after; 14239 major page faults in 15 s of generation before, 79 after;
  6.99 tokens/s before, 7.24 after.

Prompts: 55 tokens/s in 512-token batches (the GPU computes the CPU's layers
too, copying their weights over PCIe), 15 for a 16-token prompt, 28 for 64.

## Files

- `run-qwen27b.sh`: starts Qwen3.8-27B with the settings measured above
- `llama.cpp/`: the fork, branch `hybrid-moe`, on GitHub as
  [llama.cpp-hybrid-moe](https://github.com/MohammadR3za/llama.cpp-hybrid-moe)
  (git-ignored here, see `patches/`); its changes are up for review as pull request 1 there
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
