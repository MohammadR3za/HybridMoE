#!/usr/bin/env bash
# Starts Ornith-1.5-35B-A3B (the full model, Q3_K_M) on the RX 580 in hybrid mode, with the context and expert cache
# measured in README.md ("Ornith-1.5-35B-A3B"), and opens the web page once the model has loaded.
#
#   ./run-ornith35b.sh                128K context, chat page and OpenAI-compatible API on http://127.0.0.1:8080
#   CTX=256k ./run-ornith35b.sh       the model's whole context; shorter ones leave room for more experts on the GPU: 64k, 16k
#   THINK=off ./run-ornith35b.sh      answers without thinking first
#   PORT=8081 OPEN=0 ./run-ornith35b.sh another port, and no browser
#   MODEL=/path/to/model.gguf ./run-ornith35b.sh
#
# Further arguments go to llama-server as they are, after the ones below, so they win. Ctrl+C stops it.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
server=$here/llama.cpp/build/bin/llama-server
model=${MODEL:-}
if [[ -z $model ]]; then
    for dir in "$HOME/Desktop/LLMS" "$HOME/Downloads"; do
        [[ -f $dir/Ornith-1.5-35B-A3B-Q3_K_M.gguf ]] && { model=$dir/Ornith-1.5-35B-A3B-Q3_K_M.gguf; break; }
    done
fi
port=${PORT:-8080}
log=${XDG_STATE_HOME:-$HOME/.local/state}/hybridmoe/ornith35b-server.log

# The KV cache (q8_0: 2720 MiB at 256K) and the expert cache share the 8 GB, and the speed follows the size of
# the expert cache. Each pair below kept the card at 7.7-7.9 GB; tokens/s measured with them once the file
# was in the page cache: 18, 24-26, 30, 34. The first answers are slower, while the 15.5 GiB file is read.
case ${CTX:-128k} in
    256k) ctx=262144 cache=3200 ;;
    128k) ctx=131072 cache=4600 ;;
    64k)  ctx=65536  cache=5300 ;;
    16k)  ctx=16384  cache=5800 ;;
    *)    echo "CTX is 256k, 128k, 64k or 16k" >&2; exit 1 ;;
esac

[[ -x $server ]] || { echo "llama-server is not built yet: run $here/build.sh first" >&2; exit 1; }
[[ -f $model ]] || { echo "Ornith-1.5-35B-A3B-Q3_K_M.gguf is not in ~/Desktop/LLMS or ~/Downloads (set MODEL=...)" >&2; exit 1; }
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
# the others (--moe-hybrid 8 threads); with no replacements the cache keeps the experts it filled with, which
# measured 34.4 against 30.7 tokens/s in llama-bench. The experts the CPU needs must stay in the page cache, so
# the server's caches of old chats and recurrent-state snapshots are capped. MTP drafts (13 tokens/s
# against 34) did not pay off; a larger -ub costs answer speed (24-32 against 34) for a prompt that read at
# 106-187 tokens/s against 110, so pass -b 2048 -ub 2048 only for long prompts.
# The model's chat template raises "System message must be at the beginning" for the system messages that Claude Code
# adds mid-conversation, so every request of its agent loop failed with a 500. templates/ornith-agents.jinja is the
# same template with that one line rendering such a message as a user turn; it renders ordinary chats identically.
template=$here/templates/ornith-agents.jinja
[[ -f $template ]] || { echo "missing $template" >&2; exit 1; }
args=(-m "$model" -dev "$dev" --host 127.0.0.1 --port "$port" --alias ornith-35b --chat-template-file "$template"
      -ngl 99 -ncmoe 41 --moe-cache-mib "$cache" --moe-hybrid 8 --moe-hybrid-uploads 0 -t 8
      -c "$ctx" -ctk q8_0 -ctv q8_0 -fa on -np 1 --ctx-checkpoints 4 --cache-ram 1024 --agent)
case ${THINK:-on} in
    off) args+=(--reasoning off) ;;
    on)  ;;
    *)   echo "THINK is on or off" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$log")"
echo "Loading $(basename "$model") on $dev with a ${CTX:-128k} context. Full log: $log"

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
