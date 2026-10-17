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

## Long context

    ./run-ornith27b.sh    # 256K; CTX=128k, 64k or 16k leave room for more experts

Ornith's context goes up to 262,144 tokens. Only its 10 full-attention layers
keep a KV cache, of 2 heads of 256, so the whole context takes 2720 MiB in q8_0
(1440 MiB in q4_0); the other 30 layers are Gated DeltaNet, whose state does
not grow. The KV cache takes VRAM from the expert cache, and the two must leave
the card below about 7.9 GB: past that, the driver moved 580 MiB of buffers to
system memory (`mem_info_gtt_used` went from 374 to 955 MiB). At 256K with
the KV cache in q4_0, an expert cache of 5000 MiB ran at 9.7 tokens/s, and one
of 4500 MiB at 14.4-20.5; at 128K, 4900 MiB (8078 MiB of VRAM in use) at 13.1,
and 4500 MiB at 22.7.

llama-server with the KV cache in q8_0, a 256-token code answer, then a prompt
of 9650 tokens with a 128-token answer:

| Context | Expert cache | VRAM | answer | prompt | answer after it |
|---|---:|---:|---:|---:|---:|
| 16K | 5600 MiB | 7475 MiB | 24.1 | 133 | 25.4 |
| 64K | 5100 MiB | 7532 MiB | 22.9 | 123 | 23.1 |
| 128K | 4500 MiB | 7677 MiB | 22.7 | 106 | 20.7 |
| 256K | 3200 MiB | 7865 MiB | 14.9 | 88 | 17.0 |
| 256K, KV in q4_0 | 4500 MiB | 7886 MiB | 14.4, 20.5 | 83 | 16.0 |

A chat grown to 50,504 tokens in the 256K context: its first prompt took 661 s
(76 tokens/s) and the answer came at 9.0 tokens/s; the next turn processed only
its 26 new tokens (1.6 s) and answered at 12.4 tokens/s.

Runs with the same settings differ, as the 16 GB file does not fit in the
15.7 GB of RAM: the experts the CPU computes are read from the SSD when the page
cache has dropped them. Over one short and one long request, the faster runs
took 132-135 thousand major page faults for 9-10 GB (about 75 KB each); a run at
half the speed, with q8_0 keys and q4_0 values (9.5 and 8.2 tokens/s), took 1.6
million for 8.5 GB (about 5 KB each), and the 50K chat 1.6 million for 5.8 GB.
The launcher's first answer at 256K, right after loading, came at 10.0 tokens/s.

## Ornith-1.5-35B-A3B (the full model)

    ./run-ornith35b.sh    # 128K context; CTX=256k, 64k or 16k

The file is bartowski's Q3_K_M (15.55 GiB, `qwen35moe`): the model CoderX was cut
from, with all 256 experts per layer (8 used) and the same attention layout, so
its KV cache costs what Ornith's does in "Long context". It includes the MTP
head. `llama-bench` flags as in the first table; steady median of the last 4 of
10 repetitions, which were still climbing a little, as 19-31 GB were read from
the SSD per run:

| Mode | tokens/s | peak VRAM |
|---|---:|---:|
| stock split, `-ncmoe 20` / 23 | 28.1 / 30.2 | 7949 / 7535 MiB |
| hybrid, cache 5400 / **5800** / 6200 MiB, no replacements | 32.8 / **34.4** / 32.6 | 7015 / 7417 / 7819 MiB |
| hybrid, cache 5800 MiB, 1 replacement per step | 30.7 | 7417 MiB |
| hybrid, cache 5800 MiB, 6 / 10 CPU threads | 31.9 / 32.4 | 7417 MiB |

With the 5800 MiB cache, 93.8% of the selected experts run on the GPU. In
llama-server (KV in q8_0, `-ub 512`, a 256-token code answer at temperature 0,
once the file is in the page cache; the launcher's own run at 128K gave 24.0):

| Context | Expert cache | VRAM | tokens/s | 9650-token prompt |
|---|---:|---:|---:|---:|
| 16K | 5800 MiB | 7752 MiB | 34.0 | 110 tokens/s |
| 64K | 5300 MiB | 7811 MiB | 30.3 | 80 |
| 128K | 4600 MiB | 7853 MiB | 25.7 | 55 |
| 256K | 3200 MiB | 7943 MiB | 18.2 | 60 |

The speed falls as the context grows and the expert cache shrinks with it, roughly
0.6-0.8 tokens/s per 100 MiB of cache. The first
answers are slower while the experts the CPU needs are read from the SSD (code:
17.9, 31.7, then 34; chat with a new question each time: 13.8, 22.8, 26.0, 29.9),
so a new conversation starts slowly. Tried and not used, at 16K unless noted:

- **MTP drafts** (`--spec-type draft-mtp --spec-draft-n-max 3`): 13.8 tokens/s on
  code (131 of 370 drafts accepted) and 12 on chat, against 34 and 23-30 without
  (probably because the 4-token check needs up to 4 times the experts per layer;
  not measured).
- **A larger batch**: `-ub 1024` gave 30.0 tokens/s (31.8 with a 5600 MiB cache)
  and `-ub 2048` 24.1-27.4, against 34.0; the 9650-token prompt read at 106-151
  and 141-187 tokens/s against 110, which varied between runs. Pass
  `-b 2048 -ub 2048` for long prompts only.
- **KV in q4_0**: at 256K with a 4400 MiB cache 20.0 tokens/s (70 for the prompt)
  against 18.2; at 128K with 5200 MiB, 22.7 against 25.7.

### Is it smarter than CoderX?

`bench/quality/` has the tests; `qselfcheck.py` first shows that they accept
correct code and reject wrong code. The models think, at temperature 0.6, with
one sample per task and 8000 tokens at most. `qeval.py` asks 10 numeric
questions (answers computed by the script), 8 coding tasks (the model's code
runs in `bwrap`, read-only and without network, against hidden tests, some with
speed limits), and 3 instruction-following checks:

| | passed of 21 | missed |
|---|---:|---|
| Ornith-1.5-35B-A3B Q3_K_M | **20** | one coding task, out of tokens while thinking |
| Ornith-1.5-27B-A3B-CoderX Q4_K_M | 18 | digit sum of 2^100 (121, not 115), string distance, the same coding task |

`qknow.py` asks 49 factual questions with short answers (a capital, an author,
a year), at temperature 0 and without thinking (LFM2.5 always thinks):

| | correct of 49 |
|---|---:|
| Ornith-1.5-35B-A3B Q3_K_M | **49** |
| Qwen3.8-27B Q4_K_M | **49** |
| LFM2.5-24B-A2B Q4_K_M | 48 (one empty answer) |
| Ornith-1.5-27B-A3B-CoderX Q4_K_M | **22** |

CoderX, a pruned copy with 72 of 256 experts per layer removed, invents answers:
the capital of Burkina Faso, "Burkina Faso doesn't have a capital city"; the
Battle of Lepanto, "1871"; Wozzeck, "Ludwig Winter"; the neutron, "James
Neutron". The tests are small and mine, and the coding gap is two tasks of one
sample; the quiz gap is not that.

## LFM2.5-24B-A2B

    ./run-lfm25.sh    # 32K context; CTX=128k for the whole 128K

LFM2.5-24B-A2B Q4_K_M (13.5 GiB, `lfm2moe`) has 40 blocks, 2
of them dense and 38 with 64 experts of which 4 are used; 10 blocks have attention
(8 KV heads, so about 11 KiB of KV per token in q8_0, computed from the file's
head count and embedding size) and the other 30 a short convolution. Its file
fits in the 15.7 GB of RAM, so once it is in the page cache nothing is read from
the SSD: the last requests of a server read 0 MiB. Same flags as the Ornith
table (`bench/bench.py`, `llama-bench -n 128 -t 8 --repack 0 -dev Vulkan1`),
tokens/s:

| Mode | tokens/s | peak VRAM |
|---|---:|---:|
| stock split, `-ncmoe 18` / 20 / 22 / 24 / 26 | 31.7 / 32.9 / **36.4** / 33.6 / 29.0 | 7634 / 7815 / 7165 / 6467 / 5768 MiB |
| hybrid, cache 6000 MiB, no replacements | **50.1** | 6831 MiB |
| hybrid, cache 6000 MiB, 1 replacement per step | 47.7 | 6831 MiB |
| hybrid, cache 6800 MiB | 49.2 | 7636 MiB |
| hybrid, cache 7200 MiB | 47.4 | 8038 MiB |
| hybrid, cache 6800 MiB, 6 CPU threads | 50.3 | 7636 MiB |

The `-ncmoe 20` and `26` runs were still reading the file in their last
repetitions (32.0, 30.1, 33.9, 37.2 and 30.7, 32.8 tokens/s), so both are low.

    llama-bench -m LFM2.5-24B-A2B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -ncmoe 40 --moe-cache-mib 6000 --moe-hybrid 8 --moe-hybrid-uploads 0 -t 8 --repack 0

Hybrid mode runs 86% of the selected experts on the GPU with a 6000 MiB cache,
90% with 6800 and 92% with 7200 or with a replacement per step; the speed does
not rise with it (the CPU waits 0.06-0.13 ms per layer step in all of them),
and with the 7200 MiB cache, 8038 MiB of VRAM, it was lower. Prompts,
`-p 512`: 213.6 tokens/s for the stock split, 153.4 hybrid.

In llama-server (`-fa on`, KV in q8_0, `--ctx-checkpoints 4 --cache-ram 1024`,
a 256-token code answer at temperature 0) the first answers after loading run
at 14-33 tokens/s while the file is read into the page cache, and then:

| Context | Expert cache | tokens/s |
|---|---:|---:|
| 32K, 1 replacement per step | 5800 MiB | 34-36 |
| 32K, no replacements (the launcher) | 5800 MiB | 32-36 |
| 128K, no replacements (the launcher) | 5000 MiB | 26 |
| 4K, KV in f16, no replacements | 6000 MiB | 40-41 |

A 9451-token prompt was read at 149 tokens/s (32K) and 134 (128K). The server
is slower than llama-bench with the same settings, 40 against 50, as it was
with Ornith (24 against 32.6); `--no-repack` made no difference (33-36), nor did f16
instead of q8_0 for the KV cache at 32K (35-36). Not measured: the quality of
its answers.

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

### Swift-1.5's IQ3 files

[Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF)
is a fine-tune of this model whose card claims 58.5% fewer thinking tokens. Its
files mix IQ1 to IQ4 types tensor by tensor. Speed depends on the types and not
on the values, so it was measured in copies of the Q4_K_M file requantized to
the exact types of its IQ3_XXS-mtp and IQ3_S-mtp files (`--tensor-type-file`
with anchored names, and an importance matrix of ones). Code at temperature 0,
256 tokens, MTP with 3 drafts, the script's settings:

| File | GPU layers | VRAM | plain | MTP, `-t 8` | MTP, `-t 12` |
|---|---:|---:|---:|---:|---:|
| Q4_K_M, this one | 28 | 7750 MiB | 2.98 | 7.96 | - |
| IQ3_XXS-mtp layout, 10.4 GB | 43 | 7941 MiB | 6.24 | 7.56 | 8.69 |
| IQ3_S-mtp layout, 12.1 GB | 36 | 7784 MiB | 4.58 | 5.67 | 6.43 |

One token is faster (160 and 235 ms against 347), but a check of 4 tokens costs
as much as here or more (403 and 590 ms against 406, `llama-bench`): the CPU
decodes an IQ weight again for every token of a batch. On IQ3_XXS the CPU's
layers take 104 ms for 1 token and 296 ms for 4 (`GGML_SCHED_TIMING=1`). More
threads help there, unlike with Q4_K: 12 threads cut the 4-token check by 11%
and 14%, 16 and 20 no further, and the Q4_K_M's not at all (385 against 388 ms).
Drafts were accepted alike in all three (3.56-3.61 tokens per pass).

    MODEL=~/Downloads/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf NGL=43 ./run-qwen27b.sh -t 12

Not measured: its thinking length and the quality of its answers, which need
the real file.

## Files

- `run-qwen27b.sh`: starts Qwen3.8-27B with the settings measured above
- `run-ornith27b.sh`: starts Ornith-1.5-27B-A3B-CoderX in hybrid mode with a long context
- `run-lfm25.sh`: starts LFM2.5-24B-A2B in hybrid mode, 32K or 128K of context
- `run-ornith35b.sh`: starts Ornith-1.5-35B-A3B (Q3_K_M) in hybrid mode, 128K of context by default
- `bench/quality/`: head-to-head tests of two servers: reasoning, code and instruction following (`qeval.py`), facts (`qknow.py`)
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
- Reading ahead the experts the CPU is about to compute (`madvise`), so that
  the SSD reads them in large pieces and not in the 5 KB of the slow runs
  under "Long context".
