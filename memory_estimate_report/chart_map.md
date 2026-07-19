# Chart map

## Batch-1 memory reduction by preset

- Question: How does the relative benefit of blockwise attention change with model size?
- Type: Vertical bar chart, because this is one exact scalar comparison across five discrete presets.
- Encoding: Preset on x; previous/current compiler-estimate ratio on y; tooltip includes the previous and current GiB values.
- Decision use: Shows that attention dominated the small presets while parameters and optimizer state increasingly dominate the larger presets.
- Provenance: `memory_estimates.csv`, restricted to batch size 1 where both implementations were compiled directly.
