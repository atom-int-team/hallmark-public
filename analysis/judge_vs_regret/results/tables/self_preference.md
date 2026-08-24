# Confound check — judges scoring answers written by their own model

| condition | judge | n own | mean own | n other | mean other | Δ |
|---|---|---:|---:|---:|---:|---:|
| ref_based | anthropic/claude-sonnet-4.6 | 2 | 0.350 | 40 | 0.430 | -0.080 |
| ref_based | google/gemini-3.1-pro-preview | 6 | 0.700 | 36 | 0.522 | 0.178 |
| ref_based | openai/gpt-5.5 | 2 | 0.750 | 40 | 0.415 | 0.335 |
| ref_free | anthropic/claude-sonnet-4.6 | 2 | 0.400 | 40 | 0.480 | -0.080 |
| ref_free | google/gemini-3.1-pro-preview | 6 | 1.000 | 36 | 0.769 | 0.231 |
| ref_free | openai/gpt-5.5 | 2 | 0.500 | 40 | 0.525 | -0.025 |

Affects H1/H2 only; H3 is paired within a single answer.
