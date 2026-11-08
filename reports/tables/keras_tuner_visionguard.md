# Task 2.7: Keras Tuner VisionGuard Optimization (8 Trials)

|   Trial Rank |   Learning Rate |   Dropout |   Unfrozen Layers |   Val Accuracy |
|-------------:|----------------:|----------:|------------------:|---------------:|
|            1 |          0.0001 |       0.5 |                15 |         0.7808 |
|            2 |          0.0001 |       0.3 |                45 |         0.6712 |
|            3 |          1e-05  |       0.5 |                45 |         0.6233 |
|            4 |          1e-05  |       0.2 |                15 |         0.5959 |
|            5 |          1e-05  |       0.5 |                30 |         0.5822 |
|            6 |          1e-05  |       0.3 |                15 |         0.5616 |
|            7 |          0.001  |       0.3 |                30 |         0.4452 |
|            8 |          0.001  |       0.2 |                45 |         0.4452 |

### Hyperparameter Impact & Validation Overfitting Analysis:
1. **Setting That Mattered Most:** The fine-tuning learning rate was the single most decisive hyperparameter. Learning rates of 1e-3 on unfrozen layers caused gradient instability and degraded pre-trained ImageNet representations, whereas a controlled rate of 1e-5 to 1e-4 preserved foundational low-level feature detectors while allowing adaptive convergence on generative artifacts.
2. **Why Best Validation Score Can Overfit:** Iterating through numerous configurations and selecting purely by peak validation metric introduces 'hyperparameter optimization bias'. Because the validation set is finite, a configuration may score highest due to stochastic alignment with particular validation quirks rather than true underlying generalization. This is why a strictly held-out test set evaluated exactly once is vital to verify unbiased generalization.
