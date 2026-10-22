# reference — fb15k-237

Full-graph Algorithm 1: 7.70s, 8 iterations, n=14541

| local depth | ball nodes | exact B* | seconds | max abs diff to B* | mean abs diff |
|---|---|---|---|---|---|
| 1 | 2882 | False | 0.33 | 0.7580 | 0.0005 |
| 2 | 11352 | False | 1.58 | 0.3210 | 0.0000 |
| 3 | 12624 | False | 2.49 | 0.0000 | 0.0000 |
| 4 | 12798 | False | 3.34 | 0.0000 | 0.0000 |
| 5 | 12829 | False | 4.08 | 0.0000 | 0.0000 |
| 6 | 12844 | False | 4.71 | 0.0000 | 0.0000 |

Assumption 1 (eps_s): induced-subgraph labels B*_s vs full-graph B* on the same entities

| sampling | ratio | n_s | E abs(B*_s - B*) | mean B*_s | mean B* |
|---|---|---|---|---|---|
| random | 5% | 727 | 0.2278 | 0.230 | 0.002 |
| random | 10% | 1454 | 0.0749 | 0.077 | 0.002 |
| random | 20% | 2908 | 0.0221 | 0.024 | 0.002 |
| degree | 5% | 727 | 0.0161 | 0.017 | 0.000 |
| degree | 10% | 1454 | 0.0046 | 0.005 | 0.001 |
| degree | 20% | 2908 | 0.0013 | 0.002 | 0.001 |
