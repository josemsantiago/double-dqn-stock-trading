#!/usr/bin/env bash
# Rebuild the figures, the demo video, and the slide deck from the saved experiment
# outputs, then render the deck to a PDF and per-slide images for visual QA. Two runs
# produce the same deck. To regenerate the experiment outputs themselves (ten-seed
# training, the validation grid, and the cost/asset/crisis stress tests) run
#   python3 src/experiments.py --stage all
# first; it writes outputs/ and the checkpoints in models/. The written report builds
# separately with report/build_report.sh.
set -euo pipefail
cd "$(dirname "$0")"
echo "1/4  Figures...";       python3 src/plots.py
echo "2/4  Demo video...";    python3 src/demo.py
echo "3/4  Presentation...";  python3 src/build_deck.py
echo "4/4  QA render..."
mkdir -p presentation/.build
PROFILE="$(mktemp -d)"
trap 'rm -rf "$PROFILE"' EXIT
soffice --headless --convert-to pdf --outdir presentation/.build \
  -env:UserInstallation="file://$PROFILE" presentation/PRESENTATION.pptx >/dev/null 2>&1
( cd presentation/.build && rm -f slide-[0-9]*.jpg && pdftoppm -jpeg -r 120 PRESENTATION.pdf slide )
echo "Done. Deck: presentation/PRESENTATION.pptx  |  QA images: presentation/.build/slide-*.jpg"
