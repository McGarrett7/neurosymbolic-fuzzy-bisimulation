# ablation — fb15k-237

**uniform pairs**

| Method | MAE | RMSE | F1 | SPEARMAN |
|---|---|---|---|---|
| Full model | 0.500±0.001 | 0.508±0.001 | 0.183±0.054 | 0.100±0.012 |
| w/o semantic (a=1) | 0.500±0.001 | 0.508±0.001 | 0.179±0.047 | 0.100±0.012 |
| w/o graph learning (a=0) | 0.502±0.001 | 0.508±0.001 | 0.129±0.025 | 0.069±0.003 |
| w/o symbolic supervision (Jaccard target) | 0.501±0.001 | 0.507±0.001 | 0.183±0.004 | 0.096±0.009 |
| random sampling | 0.506±0.002 | 0.529±0.001 | 0.019±0.011 | 0.066±0.011 |
| with fuzzy edge weights in messages | 0.501±0.001 | 0.509±0.002 | 0.196±0.071 | 0.100±0.012 |
| paper's final ReLU (full model) | 0.500±0.002 | 0.508±0.003 | 0.132±0.031 | 0.097±0.010 |
| paper's final ReLU, w/o semantic (a=1) | 0.505±0.001 | 0.508±0.001 | 0.338±0.074 | 0.275±0.048 |

**stratified pairs**

| Method | MAE | RMSE | F1 | SPEARMAN |
|---|---|---|---|---|
| Full model | 0.455±0.011 | 0.483±0.007 | 0.876±0.041 | 0.491±0.056 |
| w/o semantic (a=1) | 0.455±0.011 | 0.482±0.007 | 0.872±0.049 | 0.491±0.056 |
| w/o graph learning (a=0) | 0.488±0.002 | 0.495±0.002 | 0.310±0.092 | 0.335±0.048 |
| w/o symbolic supervision (Jaccard target) | 0.466±0.007 | 0.485±0.004 | 0.833±0.025 | 0.474±0.043 |
| random sampling | 0.492±0.012 | 0.522±0.007 | 0.298±0.199 | 0.304±0.087 |
| with fuzzy edge weights in messages | 0.457±0.011 | 0.485±0.006 | 0.875±0.052 | 0.491±0.055 |
| paper's final ReLU (full model) | 0.462±0.006 | 0.484±0.006 | 0.823±0.024 | 0.483±0.051 |
| paper's final ReLU, w/o semantic (a=1) | 0.461±0.013 | 0.485±0.008 | 0.926±0.033 | 0.852±0.056 |

Paired t-tests vs Full model (BH-corrected p-values, stratified pairs):

- w/o semantic (a=1): mae p=7.17e-01, rmse p=7.17e-01, f1 p=7.17e-01, spearman p=7.17e-01
- w/o graph learning (a=0): mae p=2.26e-01, rmse p=2.43e-01, f1 p=1.13e-01, spearman p=1.13e-01
- w/o symbolic supervision (Jaccard target): mae p=6.36e-01, rmse p=6.36e-01, f1 p=6.36e-01, spearman p=6.36e-01
- random sampling: mae p=4.13e-01, rmse p=4.13e-01, f1 p=4.13e-01, spearman p=4.13e-01
- with fuzzy edge weights in messages: mae p=1.47e-01, rmse p=1.47e-01, f1 p=9.64e-01, spearman p=7.79e-01
- paper's final ReLU (full model): mae p=3.87e-01, rmse p=3.87e-01, f1 p=3.87e-01, spearman p=3.87e-01
- paper's final ReLU, w/o semantic (a=1): mae p=1.85e-01, rmse p=2.84e-01, f1 p=1.85e-01, spearman p=1.25e-03
