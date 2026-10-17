#!/usr/bin/env bash
# Starts Ornith-1.5-27B-A3B-CoderX on the RX 580 in hybrid mode, with the context and expert cache measured
# in README.md ("Long context"), and opens the web page once the model has loaded.
#
#   ./run-ornith27b.sh                256K context, chat page and OpenAI-compatible API on http://127.0.0.1:8080
#   CTX=128k ./run-ornith27b.sh       a shorter context leaves room for more experts on the GPU: 128k, 64k or 16k
#   THINK=off ./run-ornith27b.sh      answers without thinking first
#   PORT=8081 OPEN=0 ./run-ornith27b.sh another port, and no browser
#   MODEL=/path/to/model.gguf ./run-ornith27b.sh
#
# Further arguments go to llama-server as they are, after the ones below, so they win. Ctrl+C stops it.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
server=$here/llama.cpp/build/bin/llama-server
model=${MODEL:-}
if [[ -z $model ]]; then
    for dir in "$HOME/Desktop/LLMS" "$HOME/Downloads"; do
        [[ -f $dir/Ornith-1.5-27B-A3B-CoderX.Q4_K_M.gguf ]] && { model=$dir/Ornith-1.5-27B-A3B-CoderX.Q4_K_M.gguf; break; }
    done
fi
port=${PORT:-8080}
log=${XDG_STATE_HOME:-$HOME/.local/state}/hybridmoe/ornith27b-server.log

# The KV cache (q8_0: 2720 MiB at 256K) and the expert cache share the 8 GB. Each pair below kept the card
# under the ~7.9 GB past which the driver moved 580 MiB of buffers to system RAM, at about half the speed.
# Tokens/s measured with them, in a fresh chat: 10-17, 21-23, 23, 24. Runs differ, as the file is larger
# than the RAM: README.md, "Long context".
case ${CTX:-256k} in
    256k) ctx=262144 cache=3200 ;;
    128k) ctx=131072 cache=4500 ;;
    64k)  ctx=65536  cache=5100 ;;
    16k)  ctx=16384  cache=5600 ;;
    *)    echo "CTX is 256k, 128k, 64k or 16k" >&2; exit 1 ;;
esac

[[ -x $server ]] || { echo "llama-server is not built yet: run $here/build.sh first" >&2; exit 1; }
[[ -f $model ]] || { echo "Ornith-1.5-27B-A3B-CoderX.Q4_K_M.gguf is not in ~/Desktop/LLMS or ~/Downloads (set MODEL=...)" >&2; exit 1; }
if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "something is already listening on port $port: another server? (stop it, or set PORT=...)" >&2
    exit 1
fi

# Two models do not fit in 8 GB: say so, instead of a failed load. The RX 580 is the card with AMD's vendor id.
for card in /sys/class/drm/card[0-9]/device; do
    [[ $(cat "$card/vendor" 2>/dev/null) == 0x1002 && -r $card/mem_info_vram_used ]] || continue
    used=$(( $(cat "$card/mem_info_vram_used") >> 20 ))
    if (( used > 1024 )); then
        echo "the RX 580 already has $used MiB in use: another model is loaded (pgrep -a llama)" >&2
        exit 1
    fi
done

# The device by name, not by number: Vulkan0 is the iGPU only while the iGPU is enabled.
dev=$("$server" --list-devices 2>/dev/null | sed -n 's/^ *\(Vulkan[0-9]*\): .*RX 580.*/\1/p' | head -1)
[[ -n $dev ]] || { echo "no RX 580 among the Vulkan devices:" >&2; "$server" --list-devices >&2; exit 1; }

# -ncmoe 41 leaves every expert in host memory, from where the cache takes the hot ones, and the CPU computes
# the others (--moe-hybrid 8 threads). The file is larger than the RAM, so the experts the CPU needs come from
# the page cache or the SSD: the server's caches of old chats and recurrent-state snapshots are capped, as
# they would take that RAM.
args=(-m "$model" -dev "$dev" --host 127.0.0.1 --port "$port"
      -ngl 99 -ncmoe 41 --moe-cache-mib "$cache" --moe-hybrid 8 -t 8
      -c "$ctx" -ctk q8_0 -ctv q8_0 -fa on -np 1 --ctx-checkpoints 4 --cache-ram 1024)
case ${THINK:-on} in
    off) args+=(--reasoning off) ;;
    on)  ;;
    *)   echo "THINK is on or off" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$log")"
echo "Loading $(basename "$model") on $dev with a ${CTX:-256k} context. Full log: $log"

# The server's own output is long; the terminal shows errors, when it is ready, and each answer's speed.
"$server" "${args[@]}" "$@" 2>&1 | tee "$log" | while IFS= read -r line; do
    case $line in
        *"listening on"*)
            echo "Ready: http://127.0.0.1:$port (Ctrl+C stops the server)"
            if [[ ${OPEN:-1} != 0 && -n ${DISPLAY:-}${WAYLAND_DISPLAY:-} ]] && command -v xdg-open >/dev/null; then
                xdg-open "http://127.0.0.1:$port" >/dev/null 2>&1 || true
            fi ;;
        *"prompt eval time"*) echo "  prompt: ${line#*prompt eval time = }" ;;
        *"       eval time"*) echo "  answer: ${line#*eval time = }" ;;
        # errors: the log's E level, and what aborts print without a timestamp
        *" E "*|error:*|*"terminate called"*|*"what():"*|*GGML_ASSERT*) echo "$line" ;;
    esac
done
