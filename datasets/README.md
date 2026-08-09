# Carbon datasets

The seven-node GCP deployment expects these files in this directory:

- `Boston_24H.csv`
- `California_24H.csv`
- `South_Australia_24H.csv`
- `Nepal_24H.csv`
- `Ethiopia_24H.csv`
- `France_24H.csv`
- `Virginia_24H.csv`

CSV data is intentionally ignored by Git. Every Magellan node needs an identical copy of the full `datasets/` directory because any node can evaluate any other node as a migration destination.

Validate the directory before deployment:

```bash
python scripts/validate_seven_node_deployment.py
```
