# Benchmark Findings

Generated: 2026-04-30T10:10:25.357  Julia 1.11.3

## Setup

| Step | Time |
|------|------|
| Part 1 — schema + ACSetCategory | 991.5 ms |
| Part 2 — yoneda cache | 26175.2 ms |
| Part 2 — rules | 14518.1 ms |
| Part 3 — migrate functor | 2314.4 ms |
| Part 4 — X win-check sub-schedule | 47862.4 ms |
| Part 4 — X turn sub-schedule | 1407.9 ms |
| Part 4 — O schedules (player_migrate) | 3126.7 ms |
| Part 4 — full game schedule (baseline) | 3141.9 ms |
| Part 4 — full game schedule (cached) | 146.9 ms |

## Hom-search: baseline vs incremental cache

| Benchmark | baseline | incremental cache | speedup |
|-----------|----------|-------------------|---------|
| Part 5 — single episode (random) | 24619.8 ms | 1655.4 ms | 14.87× |
| Part 5 — 200 episodes (random) | 136.99 s | 190.99 s | 0.72× |
| Part 6 — 50 self-play episodes | 48.15 s | 47.71 s | 1.01× |
| Part 6 — full training (10×25) | 171.93 s | 238.78 s | 0.72× |

## GNN components

| Step | Time |
|------|------|
| Part 6 — world_to_gnn | 1439.22 ms |
| Part 6 — GNN forward pass | 3428.69 ms |
| Part 6 — gradient update | 42552.2 ms |
