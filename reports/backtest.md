# Backtest: train on everything before a month, pick 40 claims in that month

Validation months Jan-Jun 2026, expanding window. Only decided claims are scored (undecided cases have no outcome). Net rupees = fraud stopped - Rs 380 x genuine customers held - Rs 260 x reviews (ops-policy section 4).

## Why accuracy is the wrong KPI

| month      |   claims |   frauds |   fraud_rs |   accuracy_if_all_genuine |
|:-----------|---------:|---------:|-----------:|--------------------------:|
| 2026-01-01 |      726 |        8 |      47614 |                    0.989  |
| 2026-02-01 |      697 |        5 |      49693 |                    0.9928 |
| 2026-03-01 |      736 |        9 |      56429 |                    0.9878 |
| 2026-04-01 |      705 |        9 |      36986 |                    0.9872 |
| 2026-05-01 |      711 |       14 |      21450 |                    0.9803 |
| 2026-06-01 |      720 |       22 |      45673 |                    0.9694 |

A model that never flags anything already scores ~98-99% accuracy.

## Ranking quality per month

| month      | strategy                          |   auc |   pr_auc |   model_accuracy_at_0_5 |
|:-----------|:----------------------------------|------:|---------:|------------------------:|
| 2026-01-01 | logistic: rank by expected rupees | 0.859 |    0.565 |                  0.9917 |
| 2026-01-01 | gbm: rank by expected rupees      | 0.843 |    0.58  |                  0.9945 |
| 2026-01-01 | blend: rank by expected rupees    | 0.894 |    0.602 |                  0.9917 |
| 2026-02-01 | logistic: rank by expected rupees | 0.998 |    0.891 |                  0.9986 |
| 2026-02-01 | gbm: rank by expected rupees      | 1     |    0.967 |                  0.9986 |
| 2026-02-01 | blend: rank by expected rupees    | 1     |    1     |                  1      |
| 2026-03-01 | logistic: rank by expected rupees | 0.792 |    0.591 |                  0.9905 |
| 2026-03-01 | gbm: rank by expected rupees      | 0.84  |    0.676 |                  0.9959 |
| 2026-03-01 | blend: rank by expected rupees    | 0.791 |    0.672 |                  0.9959 |
| 2026-04-01 | logistic: rank by expected rupees | 0.701 |    0.26  |                  0.9858 |
| 2026-04-01 | gbm: rank by expected rupees      | 0.753 |    0.265 |                  0.9887 |
| 2026-04-01 | blend: rank by expected rupees    | 0.753 |    0.276 |                  0.9887 |
| 2026-05-01 | logistic: rank by expected rupees | 0.225 |    0.013 |                  0.9789 |
| 2026-05-01 | gbm: rank by expected rupees      | 0.641 |    0.031 |                  0.9775 |
| 2026-05-01 | blend: rank by expected rupees    | 0.223 |    0.013 |                  0.9761 |
| 2026-06-01 | logistic: rank by expected rupees | 0.807 |    0.32  |                  0.9597 |
| 2026-06-01 | gbm: rank by expected rupees      | 0.845 |    0.442 |                  0.9708 |
| 2026-06-01 | blend: rank by expected rupees    | 0.818 |    0.401 |                  0.9694 |

## Rupees, all six months

### Jan-Jun 2026

| strategy                                     |   reviewed |   frauds_caught |   genuine_held |   fraud_rs_stopped |   net_rs |   net_rs_per_review |   frauds_per_40 |
|:---------------------------------------------|-----------:|----------------:|---------------:|-------------------:|---------:|--------------------:|----------------:|
| gbm: rank by expected rupees                 |         55 |              25 |             30 |             152572 |   126872 |                2307 |            18.2 |
| blend: rank by expected rupees               |         59 |              25 |             34 |             153510 |   125250 |                2123 |            16.9 |
| logistic: rank by expected rupees            |         71 |              28 |             43 |             155882 |   121082 |                1705 |            15.8 |
| gbm: rank by probability                     |        240 |              36 |            204 |             165531 |    25611 |                 107 |             6   |
| blend: rank by probability                   |        240 |              34 |            206 |             163370 |    22690 |                  95 |             5.7 |
| logistic: rank by probability                |        240 |              31 |            209 |             160389 |    18569 |                  77 |             5.2 |
| baseline: 40 largest claims                  |        240 |               9 |            231 |             130671 |   -19509 |                 -81 |             1.5 |
| baseline: 40 random claims from new partners |        240 |              13 |            227 |              18075 |  -130729 |                -545 |             2.2 |
| baseline: 40 random claims                   |        240 |               2 |            238 |              15675 |  -136525 |                -569 |             0.3 |

### After the auto-approve change (May-Jun 2026)

| strategy                                     |   reviewed |   frauds_caught |   genuine_held |   fraud_rs_stopped |   net_rs |   net_rs_per_review |   frauds_per_40 |
|:---------------------------------------------|-----------:|----------------:|---------------:|-------------------:|---------:|--------------------:|----------------:|
| gbm: rank by expected rupees                 |         20 |               7 |             13 |              11441 |     1301 |                  65 |            14   |
| blend: rank by expected rupees               |         27 |               7 |             20 |              12379 |    -2241 |                 -83 |            10.4 |
| logistic: rank by expected rupees            |         39 |              11 |             28 |              17100 |    -3680 |                 -94 |            11.3 |
| gbm: rank by probability                     |         80 |              15 |             65 |              21040 |   -24460 |                -306 |             7.5 |
| blend: rank by probability                   |         80 |              15 |             65 |              21040 |   -24460 |                -306 |             7.5 |
| baseline: 40 random claims from new partners |         80 |              13 |             67 |              18075 |   -28329 |                -354 |             6.5 |
| logistic: rank by probability                |         80 |              12 |             68 |              18059 |   -28581 |                -357 |             6   |
| baseline: 40 largest claims                  |         80 |               1 |             79 |              17866 |   -32954 |                -412 |             0.5 |
| baseline: 40 random claims                   |         80 |               2 |             78 |               4577 |   -45831 |                -573 |             1   |

## Ablation: adding partner tenure ('new partner')

### Jan-Jun 2026, with tenure feature

| strategy                                     |   reviewed |   frauds_caught |   genuine_held |   fraud_rs_stopped |   net_rs |   net_rs_per_review |   frauds_per_40 |
|:---------------------------------------------|-----------:|----------------:|---------------:|-------------------:|---------:|--------------------:|----------------:|
| gbm: rank by expected rupees                 |         45 |              24 |             21 |             150434 |   130754 |                2906 |            21.3 |
| blend: rank by expected rupees               |         54 |              25 |             29 |             153367 |   128307 |                2376 |            18.5 |
| logistic: rank by expected rupees            |         75 |              31 |             44 |             160389 |   124169 |                1656 |            16.5 |
| gbm: rank by probability                     |        240 |              37 |            203 |             166370 |    26830 |                 112 |             6.2 |
| blend: rank by probability                   |        240 |              34 |            206 |             163370 |    22690 |                  95 |             5.7 |
| logistic: rank by probability                |        240 |              32 |            208 |             161429 |    19989 |                  83 |             5.3 |
| baseline: 40 largest claims                  |        240 |               9 |            231 |             130671 |   -19509 |                 -81 |             1.5 |
| baseline: 40 random claims from new partners |        240 |              13 |            227 |              18309 |  -130359 |                -543 |             2.2 |
| baseline: 40 random claims                   |        240 |               5 |            235 |              17546 |  -134496 |                -560 |             0.8 |

## Test period (Jul-Sep 2026)

Claims scored: 2252. Mean predicted fraud probability: 0.017 (expected frauds ~39). Claims with positive expected saving: 46. Review queue (<=40/month): 46 claims, expected net saving Rs 15,898.
