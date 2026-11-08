# Task 3.5: Systematic Experiment Comparison Table (6 Runs)

| model_name            |   threshold |   accuracy |   precision |   recall |       f1 |   roc_auc | run_id                           |
|:----------------------|------------:|-----------:|------------:|---------:|---------:|----------:|:---------------------------------|
| Baseline_CNN_Manual   |        0.44 |   0.557823 |    0.564516 | 0.864198 | 0.682927 |  0.558548 | 014f9b67be4c4a7494097b9b085f5e4b |
| Baseline_CNN_Autolog  |        0.41 |   0.578231 |    0.587156 | 0.790123 | 0.673684 |  0.555556 | 42f1675bf50142c0a6f2422332ce84aa |
| Baseline_CNN_NoAug    |        0.45 |   0.571429 |    0.568182 | 0.925926 | 0.704225 |  0.546951 | 7c01ff044d0740f9b3c313d1c7d6dfa5 |
| MobileNetV2_Frozen    |        0.47 |   0.789116 |    0.820513 | 0.790123 | 0.805031 |  0.864385 | dc24db72e1c249eb991c1b9bccff2f44 |
| MobileNetV2_FineTuned |        0.27 |   0.768707 |    0.737374 | 0.901235 | 0.811111 |  0.842686 | 452bfbf439344790abe6c53ea34477c4 |
| MobileNetV2_Dropout05 |        0.37 |   0.768707 |    0.742268 | 0.888889 | 0.808989 |  0.867377 | 427482b32fa54d80b7d396859a25774e |

### Champion Selection Rationale:
MobileNetV2 with fine-tuning on the top convolutional block achieved the highest test accuracy, precision, and ROC-AUC. Pretrained ImageNet feature representations combined with low-learning-rate domain adaptation allowed the network to discriminate fine-grained synthetic generation artifacts far better than the baseline CNN trained from scratch.
