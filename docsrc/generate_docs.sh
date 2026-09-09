#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")

cd "$REPO_ROOT"
uv sync --locked --extra docs
cd "$SCRIPT_DIR"
uv run make github-pages

