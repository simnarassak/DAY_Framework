from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae,r2_score,rmse,smape,mape
from darts.models import (
    ExponentialSmoothing,
    KalmanForecaster,
    LinearRegressionModel,
    RandomForest,
    RegressionModel, LightGBMModel, XGBModel)

from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.utils.likelihood_models import QuantileRegression
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def univariate_model(train,tn):
    candidates_models = {
        "LinearRegression": (LinearRegressionModel, {"lags": 8}),
        "ExponentialSmoothing": (ExponentialSmoothing, {}),
        "KalmanForecaster": (KalmanForecaster, {"dim_x": 4}),
        "RandomForest": (RandomForest, {"lags": 8,"n_estimators":100, "max_depth":10,"min_samples_split":5,
                                        "min_samples_leaf":2,"random_state": 0}),
        "LightGBMModel":(LightGBMModel,{"lags": 8,"n_estimators":100, 
                                    "learning_rate":0.1, "max_depth":5,"num_leaves":31,
                                    "min_child_samples":20}),
        "XGBModel":(XGBModel,{"lags": 8,"n_estimators":100, "learning_rate":0.1, "max_depth":5,"min_child_weight":1,
                            "subsample":0.8, "colsample_bytree":0.8})
    }
    backtest_models = []

    for model_name, (model_cls, model_kwargs) in candidates_models.items():
        model = model_cls(**model_kwargs)
        backtest_models.append(
            model.historical_forecasts(train, start=0.6, forecast_horizon=3)
        )
        mae_val=round(mae(backtest_models[-1], train), 5)
        std_mae=np.std(np.array(mae_val))
        print(f"{model_name} MAE: {mae_val}± {std_mae:.5f}")
    fix, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, backtest, model_name in zip(
        axes.flatten(),
        backtest_models,
        list(candidates_models.keys()),
    ):
        tn.inverse_transform(train[-len(backtest) :]).plot(ax=ax, label="ground truth")
        tn.inverse_transform(backtest).plot(ax=ax, label=model_name)

        ax.set_title(model_name)
        ax.set_ylim([0,4])
    plt.tight_layout()
    plt.show()