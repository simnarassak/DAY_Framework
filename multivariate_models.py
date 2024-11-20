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
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()

def multivariate_model(train,tn):
    candidates_models = {
    "MLR": (LinearRegressionModel,{"lags":8, "lags_past_covariates":8, "lags_future_covariates":[0],
                                 "add_encoders":{
                                     "cyclic": {"future": ["minute", "hour", "dayofweek", "month"]},
                                     'transformer': Scaler(scaler)}}),
    "XGB": (XGBModel,{"lags":8, "lags_past_covariates":8, "lags_future_covariates":[0],
                    "add_encoders":{
                        "cyclic": {"future": ["minute", "hour", "dayofweek", "month"]},
                        'transformer': Scaler(scaler)}}),
    "LightGBM": (LightGBMModel,{"lags":8, "lags_past_covariates":8, "lags_future_covariates":[0],
                              "add_encoders":{
                                  "cyclic": {"future": ["minute", "hour", "dayofweek", "month"]},
                                  'transformer': Scaler(scaler)}}),
    "RandomForest": (RandomForest,{"lags":8, "lags_past_covariates":8, "lags_future_covariates":[0],
                                 "add_encoders":{
                                     "cyclic": {"future": ["minute", "hour", "dayofweek", "month"]},
                                     'transformer': Scaler(scaler)}}),
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