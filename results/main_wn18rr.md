# main — wn18rr

**uniform pairs**

| Method | MAE | RMSE | F1 | SPEARMAN |
|---|---|---|---|---|
| Proposed (R-GCN + semantic, a=0.7) | 0.557±0.007 | 0.586±0.011 | 0.008±0.006 | 0.099±0.006 |
| R-GCN | 0.556±0.006 | 0.585±0.009 | 0.007±0.006 | 0.098±0.007 |
| GCN | 0.536±0.006 | 0.559±0.008 | 0.008±0.006 | 0.085±0.012 |
| GraphSAGE | 0.552±0.008 | 0.579±0.009 | 0.007±0.005 | 0.093±0.006 |
| MLP on structural features | 0.537±0.004 | 0.556±0.007 | 0.012±0.007 | 0.069±0.018 |
| Semantic only (trained projection) | 0.519±0.001 | 0.529±0.003 | 0.002±0.003 | 0.037±0.011 |

**stratified pairs**

| Method | MAE | RMSE | F1 | SPEARMAN |
|---|---|---|---|---|
| Proposed (R-GCN + semantic, a=0.7) | 0.548±0.011 | 0.580±0.014 | 0.117±0.096 | 0.293±0.034 |
| R-GCN | 0.547±0.010 | 0.579±0.012 | 0.115±0.094 | 0.292±0.035 |
| GCN | 0.527±0.009 | 0.553±0.010 | 0.103±0.064 | 0.250±0.017 |
| GraphSAGE | 0.541±0.005 | 0.573±0.009 | 0.111±0.081 | 0.283±0.029 |
| MLP on structural features | 0.526±0.003 | 0.548±0.006 | 0.158±0.092 | 0.232±0.041 |
| Semantic only (trained projection) | 0.513±0.005 | 0.524±0.007 | 0.026±0.026 | 0.114±0.030 |

Paired t-tests vs Proposed (R-GCN + semantic, a=0.7) (BH-corrected p-values, stratified pairs):

- R-GCN: mae p=4.53e-01, rmse p=4.53e-01, f1 p=4.53e-01, spearman p=4.53e-01
- GCN: mae p=8.47e-02, rmse p=8.47e-02, f1 p=6.31e-01, spearman p=1.05e-01
- GraphSAGE: mae p=5.10e-01, rmse p=5.10e-01, f1 p=6.55e-01, spearman p=5.06e-01
- MLP on structural features: mae p=7.33e-02, rmse p=7.33e-02, f1 p=7.33e-02, spearman p=7.33e-02
- Semantic only (trained projection): mae p=1.52e-02, rmse p=1.51e-02, f1 p=2.21e-01, spearman p=4.60e-03
