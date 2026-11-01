# Stage 4D.3 u75 calibrated replay animation

`scripts/animate_stage4d3_u75.py` turns one recorded Stage 4D.3 `u75`
scenario into a world-map animation. It is deliberately labeled **calibrated
replay — not live execution**.

The default is the canonical summer scenario,
`summer-20240820T1200Z-u75`, under
`magellan_capacity_lowest_score`. This representative scenario has the actual
nine-task `u75` layout (three benchmark, three Dendro-GR, and three LLM tasks)
and seven accepted migrations. The script reads rather than hard-codes task
IDs, placements, timestamps, and routes. It verifies `checksums.sha256` and
requires every row in `migration_events.csv` to have a matching `accepted` row
in `auction_events.csv`.

## Install

From the Magellan V2 repository root:

```bash
python3 -m venv .venv-animation
source .venv-animation/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[visualization,cartography]'
```

MP4 export also needs `ffmpeg` on `PATH`. On macOS with Homebrew:

```bash
brew install ffmpeg
```

The recommended command above installs Cartopy and renders a Robinson world map
with Natural Earth coastlines. If Cartopy cannot be installed, use the smaller
fallback environment:

```bash
python -m pip install -e '.[visualization]'
```

Cartopy downloads its low-resolution Natural Earth coastline data on first use.
If that is unavailable, rendering automatically falls back to the offline
schematic map. Use `--no-cartopy` to select that fallback explicitly.

## Validate and render the canonical bundle

Set the bundle path once (the bundle itself remains outside Git):

```bash
BUNDLE=/Users/WILL/Downloads/experiments/measurements/stage4d3-20260831T231409Z-ff3d35b5
python scripts/animate_stage4d3_u75.py --bundle "$BUNDLE" --validate-only
python scripts/animate_stage4d3_u75.py --bundle "$BUNDLE" --gif
```

The default outputs are:

```text
experiments/visualizations/stage4d3-u75-summer-calibrated-replay.mp4
experiments/visualizations/stage4d3-u75-summer-calibrated-replay.gif
```

The output directory is ignored by Git so rendered media does not enter a paper
commit accidentally.
To choose explicit paths or render another recorded season:

```bash
python scripts/animate_stage4d3_u75.py \
  --bundle "$BUNDLE" \
  --season winter \
  --output /tmp/stage4d3-u75-winter.mp4 \
  --gif /tmp/stage4d3-u75-winter.gif
```

Quiet periods are compressed to keep the visualization short. During each
migration transition, the displayed UTC clock advances from the recorded
`started_at_utc` to `finished_at_utc`; the route and task movement are not
invented. Site labels remain fixed, colocated tasks receive deterministic
offsets, and completed arcs fade after arrival. The small CPU annotation beside
each site comes from `occupancy_timeline.csv` and is cross-checked against
`config/cluster.gcp.json`; hide it with `--hide-capacity`.
