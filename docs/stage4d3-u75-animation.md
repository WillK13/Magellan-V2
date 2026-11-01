# Stage 4D.3 calibrated replay animation

`scripts/animate_stage4d3_u75.py` turns a recorded Stage 4D.3 capacity scenario
into a world-map animation. It is deliberately labeled **calibrated replay —
not live execution**.

The default is the canonical summer scenario,
`summer-20240820T1200Z-u75`, under
`magellan_capacity_lowest_score`. This representative scenario has the actual
nine-task `u75` layout (three benchmark, three Dendro-GR, and three LLM tasks)
and seven accepted migrations. The script reads rather than hard-codes task
IDs, placements, timestamps, and routes. It verifies `checksums.sha256` and
requires every row in `migration_events.csv` to have a matching `accepted` row
in `auction_events.csv`.

Use `--load-id umax` for the busiest canonical case: the maximal measured
packing of 11 tasks (4 benchmark, 3 Dendro-GR, and 4 LLM). The renderer also
supports `u25`, `u50`, and `u75`. Completed tasks disappear at their recorded
completion times so the map shows current rather than cumulative placement.
Each site halo follows recorded CPU occupancy: cyan below 75%, yellow from
75–95%, and red at 95% or greater. The side panel also reports the number of
final sites under the capacity policy versus the bundle's unlimited reference;
for summer `umax`, the recorded outcomes are spread over four sites instead of
all 11 tasks selecting the same unlimited-reference site.

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

Render and preview the busiest capacity-constrained summer replay:

```bash
python scripts/animate_stage4d3_u75.py \
  --bundle "$BUNDLE" \
  --load-id umax \
  --gif

open experiments/visualizations/magellan-replay-visualization-umax-summer.mp4
```

The default outputs are:

```text
experiments/visualizations/magellan-replay-visualization-u75-summer.mp4
experiments/visualizations/magellan-replay-visualization-u75-summer.gif
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
offsets, completed tasks disappear, and completed arcs fade after arrival. The
CPU annotation and utilization halo at each site come from
`occupancy_timeline.csv` and are cross-checked against
`config/cluster.gcp.json`; hide them with `--hide-capacity`.
