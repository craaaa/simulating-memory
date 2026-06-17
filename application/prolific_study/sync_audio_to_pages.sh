#!/usr/bin/env bash
# Copies birds_texts/*.wav to the gh-pages branch and pushes.
# Run from anywhere inside the repo after generating new wav files.
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
WAV_SRC="$REPO_ROOT/application/prolific_study/birds_texts"
PAGES_WORKTREE="$REPO_ROOT/.git/pages-worktree"

# Create a temporary worktree on gh-pages if not already present.
if [ ! -d "$PAGES_WORKTREE" ]; then
    git -C "$REPO_ROOT" worktree add "$PAGES_WORKTREE" gh-pages
fi

DEST="$PAGES_WORKTREE/birds_texts"
mkdir -p "$DEST"

# Copy only wav files.
cp "$WAV_SRC"/*.wav "$DEST/"
echo "Copied $(ls "$DEST"/*.wav | wc -l | tr -d ' ') wav files to gh-pages worktree."

# Commit and push.
git -C "$PAGES_WORKTREE" add birds_texts/*.wav
if git -C "$PAGES_WORKTREE" diff --cached --quiet; then
    echo "No changes — gh-pages already up to date."
else
    git -C "$PAGES_WORKTREE" commit -m "Sync audio stimuli $(date +%Y%m%d)"
    git -C "$PAGES_WORKTREE" push origin gh-pages
    echo "Pushed to gh-pages."
fi
