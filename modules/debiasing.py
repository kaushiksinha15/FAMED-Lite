from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import equalized_odds_difference

def train_fairness_optimizer(meta_learner, meta_X, y, sensitive_features):
    """
    Train a fairness optimizer using Fairlearn.
    """
    optimizer = ThresholdOptimizer(
        estimator=meta_learner,
        constraints="equalized_odds",
        predict_method="predict",
        prefit=True
    )
    optimizer.fit(meta_X, y, sensitive_features=sensitive_features)
    return optimizer

def calculate_disparity(y_true, y_pred, sensitive_features):
    """
    Calculate equalized odds difference to quantify bias.
    """
    return equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_features)
