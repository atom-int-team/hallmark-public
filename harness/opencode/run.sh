#!/bin/sh
# hallmark-opencode <model-slug> <variant> <prompt>
#
# Container entry point of the opencode arm. It exists for one reason: small_model must
# equal the cell's model (otherwise the title agent calls a different model), and opencode
# config is a file, not a flag. The wrapper adds the field to the base config and execs
# opencode with pinned flags:
#   --format json  event stream on stdout;  --pure  no external plugins;
#   --thinking     reasoning blocks in the stream (for the solution trace);
#   --print-logs   stderr with modelID and 'stream error' — the adapter uses these to tell
#                  a transport fault from agent silence (stdout carries almost no errors).
set -eu
MODEL="$1"; VARIANT="$2"; PROMPT="$3"
python3 - "$MODEL" <<'PY'
import json, pathlib, sys
cfg_dir = pathlib.Path.home() / ".config" / "opencode"
cfg = json.loads((cfg_dir / "opencode.base.json").read_text())
cfg["small_model"] = f"openrouter/{sys.argv[1]}"
(cfg_dir / "opencode.json").write_text(json.dumps(cfg))
PY
# The prompt goes through stdin, NOT as a positional argument: `opencode run [message..]`
# joins positionals as `G.includes(" ") ? '"' + G.replace(/"/g,'\\"') + '"' : G`,
# i.e. wraps the text in quotes and escapes every inner quote (the model would see
# \"Final: …\" inside quotes). From a non-TTY stdin the text is taken verbatim.
printf '%s' "$PROMPT" | exec opencode run --format json --pure --thinking --print-logs \
     --log-level INFO -m "openrouter/$MODEL" --variant "$VARIANT"
