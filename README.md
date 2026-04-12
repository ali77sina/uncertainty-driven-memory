# Uncertainty-Driven Memory for Long-Horizon Transformers

Comparing three memory strategies for transformers on long-horizon tasks:

1. **Baseline**: standard transformer with truncated context window
2. **RMT**: Recurrent Memory Transformer with FIFO memory tokens
3. **Uncertainty Memory**: memory tokens with uncertainty-gated writes (ours)

Tested on synthetic bAbI-style tasks with increasing story length.

## Run

```bash
pip install -r requirements.txt
python experiment.py
```

Figures are saved to `figures/`.
