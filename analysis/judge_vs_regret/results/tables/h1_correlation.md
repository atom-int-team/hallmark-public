# H1 — judge score vs objective score (original answers)

| condition | judge | n | Pearson | p | Spearman | p | mean judge | mean objective |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ref_based | anthropic/claude-sonnet-4.6 | 42 | 0.784 | 0.000 | 0.769 | 0.000 | 0.426 | 0.472 |
| ref_based | google/gemini-3.1-pro-preview | 42 | 0.843 | 0.000 | 0.799 | 0.000 | 0.548 | 0.472 |
| ref_based | openai/gpt-5.5 | 42 | 0.832 | 0.000 | 0.778 | 0.000 | 0.431 | 0.472 |
| ref_based | POOLED | 42 | 0.848 | 0.000 | 0.794 | 0.000 | 0.468 | 0.472 |
| ref_free | anthropic/claude-sonnet-4.6 | 42 | 0.406 | 0.008 | 0.387 | 0.011 | 0.476 | 0.472 |
| ref_free | google/gemini-3.1-pro-preview | 42 | 0.522 | 0.000 | 0.581 | 0.000 | 0.802 | 0.472 |
| ref_free | openai/gpt-5.5 | 42 | 0.559 | 0.000 | 0.534 | 0.000 | 0.524 | 0.472 |
| ref_free | POOLED | 42 | 0.567 | 0.000 | 0.529 | 0.000 | 0.601 | 0.472 |
