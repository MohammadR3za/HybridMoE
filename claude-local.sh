#!/usr/bin/env bash
# Runs Claude Code on the local model that run-ornith35b.sh serves, and leaves your normal Claude Code alone: the
# settings below apply to this one process, and it keeps its own config folder (~/.claude-local), so it has none of your
# skills, plugins or MCP servers, which would only make the model's prompt longer.
#
#   ./run-ornith35b.sh            in one terminal (serves http://127.0.0.1:8080)
#   ./claude-local.sh             in another, from the project you want to work on
#   ./claude-local.sh -p "..."    one question, no screen
#
#   LOCAL_URL=http://127.0.0.1:8081 ./claude-local.sh      another port
#   CLAUDE_LOCAL_TOOLS="Read,Edit,Bash,Glob,Grep,WebFetch" ./claude-local.sh   other built-in tools
#
# Anthropic does not support Claude Code with other models; this works through the Anthropic API that llama-server
# offers. Why each setting is here is in README.md ("Claude Code and OpenCode").
set -euo pipefail

url=${LOCAL_URL:-http://127.0.0.1:8080}
props=$(curl -sf --max-time 5 "$url/props") || { echo "no llama-server at $url: start ./run-ornith35b.sh first" >&2; exit 1; }
ctx=$(python3 -I -c 'import json,sys; print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])' <<<"$props")
model=$(curl -sf --max-time 5 "$url/v1/models" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')

unset ANTHROPIC_API_KEY
export CLAUDE_CONFIG_DIR=${CLAUDE_LOCAL_DIR:-$HOME/.claude-local}
export ANTHROPIC_BASE_URL=$url ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=$model
# Claude Code assumes 200K for a model it does not know; the server's real window avoids a failed request near the end.
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=$ctx
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_ATTRIBUTION_HEADER=0 CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1

# Six tools instead of all of them: the first request is 4.7K tokens instead of 15.7K (61 s instead of 193 s to read).
# The second flag moves the folder and git status out of the system prompt, so the start of the prompt is the same in
# every session and the server reuses it: a second session needed 926 new tokens (16 s) instead of 4.7K.
exec claude --exclude-dynamic-system-prompt-sections --tools "${CLAUDE_LOCAL_TOOLS:-Read,Write,Edit,Bash,Glob,Grep}" "$@"
