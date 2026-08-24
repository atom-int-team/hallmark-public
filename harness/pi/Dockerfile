# Sandbox for the harness arm: pi + scientific stack, no outbound network.
#
# This file sets the harness's capability ceiling — the package list is reported in the
# paper. Changing the list changes the system: build a new image version under a new
# system id, never silently redefine an existing arm.
#
# Build:  docker build --build-arg PI_VERSION=0.84.3 -t hallmark-harness:pi-0.84.3 harness/pi
FROM python:3.12-slim-bookworm

# Node comes from the official image — bookworm apt ships one that is too old.
COPY --from=node:24-bookworm-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:24-bookworm-slim /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

# Utilities the agent actually uses through bash (e.g. head/wc/python3).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates coreutils grep sed gawk less procps \
 && rm -rf /var/lib/apt/lists/*

# Scientific stack, installed FROM THE LOCK (requirements.lock, built from requirements.txt):
# exact versions are part of the system definition; changing them means a new arm.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock && rm /tmp/requirements.lock

ARG PI_VERSION
RUN npm install -g --ignore-scripts "@earendil-works/pi-coding-agent@${PI_VERSION}" \
 && pi --version

# The agent is not root and sees only /work (the repository is never mounted).
RUN useradd -m -u 1000 agent && mkdir -p /home/agent/.pi/agent
COPY models.json /home/agent/.pi/agent/models.json
RUN chown -R agent:agent /home/agent
USER agent

# No startup network operations (version/catalog checks) — the network is closed.
ENV PI_OFFLINE=1 \
    HOME=/home/agent \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /work
