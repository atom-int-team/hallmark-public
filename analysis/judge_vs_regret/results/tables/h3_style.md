# H3/H4 — style effect on the judge, with the objective grader as control

| condition | judge | n | mean Δ(verbose−terse) | 95% CI | median | Wilcoxon p | share Δ>0 |
|---|---|---:|---:|---|---:|---:|---:|
| ref_based | anthropic/claude-sonnet-4.6 | 42 | 0.105 | [0.074, 0.138] | 0.100 | 0.000 | 0.67 |
| ref_based | google/gemini-3.1-pro-preview | 42 | 0.086 | [0.033, 0.143] | 0.000 | 0.010 | 0.48 |
| ref_based | openai/gpt-5.5 | 42 | 0.031 | [0.010, 0.052] | 0.000 | 0.013 | 0.38 |
| ref_based | POOLED | 42 | 0.074 | [0.046, 0.105] | 0.067 | 0.000 | 0.74 |
| ref_free | anthropic/claude-sonnet-4.6 | 42 | 0.240 | [0.198, 0.286] | 0.200 | 0.000 | 0.93 |
| ref_free | google/gemini-3.1-pro-preview | 42 | 0.367 | [0.271, 0.462] | 0.400 | 0.000 | 0.83 |
| ref_free | openai/gpt-5.5 | 42 | 0.183 | [0.131, 0.238] | 0.100 | 0.000 | 0.76 |
| ref_free | POOLED | 42 | 0.263 | [0.210, 0.320] | 0.267 | 0.000 | 0.86 |

Objective score, same texts: max |Δ| = 0.0e+00 over 84 rewrites.
