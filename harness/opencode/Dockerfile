# Sandbox for the second harness of the harness axis: opencode + THE SAME scientific stack
# as pi, no outbound network.
#
# requirements.lock is a byte-for-byte copy of harness/pi/requirements.lock: the scientific
# stack is a property of the AXIS (the capability ceiling of any harness), not of an arm;
# tests/test_cli_agent.py::test_opencode_lock_is_identical_to_pi keeps them equal.
# Changing the list changes the axis: new versions of BOTH images and new system ids.
#
# Build:  docker build --build-arg OPENCODE_VERSION=1.18.29 \
#              -t hallmark-harness:opencode-1.18.29-cfg2 harness/opencode   (cfg2 = opencode.json with external_directory: allow)
FROM python:3.12-slim-bookworm

# Node is needed only by the opencode npm installer (opencode itself is a bun binary).
COPY --from=node:24-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:24-bookworm-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Same utilities as in the pi image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates coreutils grep sed gawk less procps \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock && rm /tmp/requirements.lock

# The opencode-ai postinstall fetches the platform package (opencode-linux-arm64/x64) — the
# network is available at build time; at runtime the binary downloads nothing (see ENV below).
ARG OPENCODE_VERSION=1.18.29
RUN npm install -g "opencode-ai@${OPENCODE_VERSION}" && opencode --version

# The agent is not root and sees only /work. HOME lives in the image layer and the container
# is --rm: opencode's sqlite session DB dies with the container — a cell never sees the
# previous one (the analogue of pi's --no-session).
RUN useradd -m -u 1000 agent && mkdir -p /home/agent/.config/opencode
COPY opencode.json /home/agent/.config/opencode/opencode.base.json
COPY run.sh /usr/local/bin/hallmark-opencode
RUN chmod +x /usr/local/bin/hallmark-opencode && chown -R agent:agent /home/agent
USER agent

# No startup network operations and no pickup of host instructions.
# OPENCODE_DISABLE_CLAUDE_CODE: by default opencode reads Claude Code's user-level CLAUDE.md
# and skills — they do not exist in the container, but the flag is part of the arm definition.
ENV HOME=/home/agent \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCODE_DISABLE_CLAUDE_CODE=1 \
    OPENCODE_DISABLE_MODELS_FETCH=1 \
    OPENCODE_DISABLE_AUTOUPDATE=1 \
    OPENCODE_DISABLE_LSP_DOWNLOAD=1 \
    OPENCODE_DISABLE_EXTERNAL_SKILLS=1 \
    OPENCODE_DISABLE_DEFAULT_PLUGINS=1

WORKDIR /work
ENTRYPOINT ["hallmark-opencode"]
