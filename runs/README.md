# runs/

Each `*.json` here is one full bench result, captured by `bench.sh`.
Filenames are `<UTC-timestamp>_<safe-label>.json`.

Contents per file:

```json
{
  "timestamp": "...",
  "label": "...",
  "env": { "chip": "...", "memory_gb": 64, "cores": 12, ... },
  "raw":     [ /* one entry per probe run */ ],
  "summary": [ /* min/median/max per probe */ ]
}
```

To submit a result: commit your JSON and open a PR.
