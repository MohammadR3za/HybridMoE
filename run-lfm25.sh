#!/usr/bin/env bash
# Starts LFM2.5-24B-A2B on the RX 580 in hybrid mode, with the context and expert cache measured in
# README.md ("LFM2.5-24B-A2B"), and opens the web page once the model has loaded.
#
#   ./run-lfm25.sh                32K context, chat page and OpenAI-compatible API on http://127.0.0.1:8080
#   CTX=128k ./run-lfm25.sh       the model's whole context, with a smaller expert cache
#   PORT=8081 OPEN=0 ./run-lfm25.sh   another port, and no browser
#   MODEL=/path/to/model.gguf ./run-lfm25.sh
#
# The first answers are slow (14-33 tokens/s): the 13.5 GiB file is read into the page cache as the CPU
# needs its experts, and from about the fifth answer on nothing is read from the SSD.
# Further arguments go to llama-server as they are, after the ones below, so they win. Ctrl+C stops it.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
server=$here/llama.cpp/build/bin/llama-server
model=${MODEL:-$HOME/Downloads/LFM2.5-24B-A2B-Q4_K_M.gguf}
port=${PORT:-8080}
log=${XDG_STATE_HOME:-$HOME/.local/state}/hybridmoe/lfm25-server.log

# The KV cache (q8_0: about 11 KiB per token) and the expert cache share the 8 GB. The pair for each context
# kept the card at 7.0-7.4 GB; past about 7.9 GB the driver moves buffers to system RAM and speed drops.
# Tokens/s measured with them once the file was in the page cache: 34-36 and 26.
case ${CTX:-32k} in
    32k)  ctx=32768  cache=5800 ;;
    128k) ctx=131072 cache=5000 ;;
    *)    echo "CTX is 32k or 128k" >&2; exit 1 ;;
esac

[[ -x $server ]] || { echo "llama-server is not built yet: run $here/build.sh first" >&2; exit 1; }
[[ -f $model ]] || { echo "model not found: $model (set MODEL=...)" >&2; exit 1; }
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

# -ncmoe 40 leaves every expert in host memory, from where the cache takes the hot ones, and the CPU computes
# the others (--moe-hybrid 8 threads); with no replacements the cache keeps the experts it filled with, which
# measured the same in the server and 5% faster in llama-bench. The server's caches of old chats and
# recurrent-state snapshots are capped, as they would take the RAM that holds the model file.
args=(-m "$model" -dev "$dev" --host 127.0.0.1 --port "$port"
      -ngl 99 -ncmoe 40 --moe-cache-mib "$cache" --moe-hybrid 8 --moe-hybrid-uploads 0 -t 8
      -c "$ctx" -ctk q8_0 -ctv q8_0 -fa on -np 1 --ctx-checkpoints 4 --cache-ram 1024)

mkdir -p "$(dirname "$log")"
echo "Loading $(basename "$model") on $dev with a ${CTX:-32k} context. Full log: $log"

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
