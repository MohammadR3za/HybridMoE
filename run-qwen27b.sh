#!/usr/bin/env bash
# Starts Qwen3.8-27B on the RX 580 with the settings measured in README.md ("Qwen3.8-27B"), and opens the
# web page once the model has loaded. Loading takes about 40 s, and until then nothing answers on the port.
#
#   ./run-qwen27b.sh                  chat page and OpenAI-compatible API on http://127.0.0.1:8080
#   EFFORT=medium ./run-qwen27b.sh    shorter thinking: low, medium or xhigh (the model's default)
#   THINK=off ./run-qwen27b.sh        answers without thinking first
#   PORT=8081 OPEN=0 ./run-qwen27b.sh another port, and no browser
#   MODEL=/path/to/model.gguf ./run-qwen27b.sh
#   NGL=34 MODEL=.../Qwen3.8-27B-Q3_K_M.gguf ./run-qwen27b.sh   a smaller file has room for more GPU layers
#
# Further arguments go to llama-server as they are, after the ones below, so they win. Ctrl+C stops it.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
server=$here/llama.cpp/build/bin/llama-server
model=${MODEL:-$HOME/Downloads/Qwen3.8-27B-Q4_K_M.gguf}
port=${PORT:-8080}
log=${XDG_STATE_HOME:-$HOME/.local/state}/hybridmoe/qwen27b-server.log

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

# The 9 GB of weights the CPU reads are file pages, and must stay in the 15.7 GB of RAM, or tokens wait on
# the SSD. The server's defaults would let it take up to 8 GB for old chats (--cache-ram) and 32 x 150 MiB
# for snapshots of the recurrent state (--ctx-checkpoints), so both are capped.
#
# 28 GPU layers are the output head and the last 27 blocks, the MTP one among them: 7.75 GB of the 8 GB with
# the context below. 29 and 30 were slower (7.04 and 6.66 tokens/s against 7.79), the card being full; a
# Q3_K_M file fits 34 (NGL=34). The context is 16K, since thinking at the default effort fills 8K in a few
# answers: its KV cache in q8_0 takes the room of an 8K one in f16, at the same speed, and the draft's small
# ubatch saves 134 MiB of VRAM.
args=(-m "$model" -dev "$dev" --host 127.0.0.1 --port "$port"
      -ngl "${NGL:-28}" -c 16384 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 8
      --spec-type draft-mtp --spec-draft-n-max 3 -ctkd q8_0 -ctvd q8_0 --spec-draft-n-ubatch 16
      --ctx-checkpoints 4 --cache-ram 1024)
case ${THINK:-on} in
    off) args+=(--reasoning off) ;;
    on)  [[ -n ${EFFORT:-} ]] && args+=(--reasoning-effort "$EFFORT") ;;
    *)   echo "THINK is on or off" >&2; exit 1 ;;
esac

mkdir -p "$(dirname "$log")"
echo "Loading $(basename "$model") on $dev, about 40 s. Full log: $log"

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
        *"draft acceptance"*) echo "  drafts: ${line#*draft acceptance = }" ;;
        # errors: the log's E level, and what aborts print without a timestamp
        *" E "*|error:*|*"terminate called"*|*"what():"*|*GGML_ASSERT*) echo "$line" ;;
    esac
done
