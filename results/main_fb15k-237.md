# main — fb15k-237

**uniform pairs**

| Method | MAE | RMSE | F1 | SPEARMAN |
|---|---|---|---|---|
| Proposed (R-GCN + semantic, a=0.7) | 0.501±0.001 | 0.508±0.001 | 0.151±0.058 | 0.085±0.024 |
| R-GCN | 0.500±0.001 | 0.508±0.001 | 0.148±0.061 | 0.085±0.024 |
| GCN | 0.504±0.001 | 0.512±0.001 | 0.045±0.034 | 0.042±0.034 |
| GraphSAGE | 0.502±0.001 | 0.510±0.002 | 0.109±0.074 | 0.076±0.034 |
| MLP on structural features | 0.504±0.002 | 0.513±0.003 | 0.082±0.055 | 0.062±0.019 |
| Semantic only (trained projection) | 0.502±0.001 | 0.508±0.001 | 0.086±0.064 | 0.053±0.022 |

**stratified pairs**

| Method | MAE | RMSE | F1 | SPEARMAN |
|---|---|---|---|---|
| Proposed (R-GCN + semantic, a=0.7) | 0.464±0.015 | 0.487±0.008 | 0.852±0.050 | 0.441±0.084 |
| R-GCN | 0.464±0.016 | 0.487±0.009 | 0.840±0.060 | 0.441±0.084 |
| GCN | 0.492±0.007 | 0.504±0.006 | 0.353±0.110 | 0.235±0.133 |
| GraphSAGE | 0.471±0.014 | 0.492±0.009 | 0.719±0.132 | 0.398±0.114 |
| MLP on structural features | 0.478±0.012 | 0.498±0.007 | 0.628±0.073 | 0.337±0.081 |
| Semantic only (trained projection) | 0.491±0.005 | 0.498±0.005 | 0.251±0.112 | 0.281±0.086 |

Paired t-tests vs Proposed (R-GCN + semantic, a=0.7) (BH-corrected p-values, stratified pairs):

- R-GCN: mae p=7.01e-01, rmse p=7.01e-01, f1 p=7.01e-01, spearman p=7.01e-01
- GCN: mae p=3.68e-02, rmse p=2.97e-02, f1 p=2.97e-02, spearman p=3.63e-02
- GraphSAGE: mae p=2.38e-01, rmse p=2.61e-01, f1 p=2.61e-01, spearman p=2.61e-01
- MLP on structural features: mae p=2.65e-02, rmse p=2.21e-02, f1 p=2.21e-02, spearman p=2.70e-02
- Semantic only (trained projection): mae p=8.29e-02, rmse p=8.29e-02, f1 p=1.13e-02, spearman p=5.51e-03
