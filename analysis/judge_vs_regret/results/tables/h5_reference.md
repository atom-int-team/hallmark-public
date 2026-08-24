# H5 — does giving the judge the ground truth rescue it?

| condition | metric | value |
|---|---|---:|
| ref_based | Spearman(judge, objective) | 0.794 |
| ref_based | judge mean on objective=0 | 0.197 |
| ref_based | share judge≥0.9 on objective=0 | 0.00 |
| ref_based | mean Δ(verbose−terse) | 0.074 |
| ref_free | Spearman(judge, objective) | 0.529 |
| ref_free | judge mean on objective=0 | 0.453 |
| ref_free | share judge≥0.9 on objective=0 | 0.00 |
| ref_free | mean Δ(verbose−terse) | 0.263 |
