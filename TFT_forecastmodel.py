import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.utils.likelihood_models import QuantileRegression
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
from darts.metrics import mae,r2_score,rmse,smape,mape
from darts.models import TFTModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models import GaussianLikelihood


def tft_forecast(train1,train2,test1,test2,tn,covs1,covs2):
    
    quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,]
    
    input_chunk_length = 20
    forecast_horizon = 9
    TFT_forecast_model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=forecast_horizon,
    hidden_size=32,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=16,
    n_epochs=100,
    optimizer_kwargs={"lr": 1e-3},
    add_relative_index=False,
    add_encoders={"cyclic": {"future": ["month"]}},
    likelihood=QuantileRegression(
        quantiles=quantiles
    ),  
    random_state=42,)
    
    
    #SOM model
    #TFT1=TFT_forecast_model.fit(series=train1,verbose=True)
    #SOMEV model
    #TFT2=TFT_forecast_model.fit(series=train1,future_covariates=covs1,verbose=True)
    #MOM model
    #TFT3=TFT_forecast_model.fit(series=[train1,train2],verbose=True)
    
    #MOMEV model
    TFT4=TFT_forecast_model.fit(series=[train1,train2],future_covariates=[covs1,covs2],verbose=True)
    
    # prediction
    pred_TFT4 = TFT4.predict(n=14, series=train1,future_covariates=covs1,verbose=True)
    
    
    pr1=tn.inverse_transform(pred_TFT4)
    df_tft1=pr1.pd_dataframe()
    df_tft1=df_tft1.reset_index()
    df_tft1=df_tft1.rename(columns={'y': 'MOMEV'})
    
    tdf=tn.inverse_transform(test1)
    test_df=tdf.pd_dataframe()
    test_df=test_df.reset_index()
    test_df=test_df.rename(columns={'y': 'Observed Yield'})
    
    TFT_prediction_df=test_df.merge(df_tft1,how="left",on="ds")
    # plot
    plt.figure(figsize=(12, 8))

    # Plot the original data
    plt.plot(TFT_prediction_df.ds,TFT_prediction_df['Observed Yield'],label='Observed Yield',color='black')

    # Plot model predictions
    plt.plot(TFT_prediction_df.ds,TFT_prediction_df.MOMEV, label='MOMEV', linestyle='--')

    plt.legend()

    plt.grid(True, alpha=0.5)
    plt.figure(figsize=(12, 4))

    # Calculate differences
    diff4 = TFT_prediction_df['Observed Yield']- TFT_prediction_df.MOMEV

    # Plot differences
    plt.plot(TFT_prediction_df.ds, diff4, label='MOMEV', color='orange')

    plt.axhline(0, color='black', linestyle='--')  # Zero line

    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.legend()

    plt.grid(True, alpha=0.5)
    
    '''
    # sliding window prediction
    
    def windowed_predict(model, series, future_covariates, test_length, window_size=14, gap=30):
    predictions = []
    current_index = 0

    while current_index < test_length:
        # Predict the next window
        window_pred = model.predict(
            n=window_size,
            series=series,
            future_covariates=future_covariates
        )
        
        # Add the predictions to our list
        predictions.append(window_pred)
        
        # Move the index forward
        current_index += window_size + gap

        # Update the series with the new predictions
        series = pd.concat([series, window_pred])

    # Concatenate all predictions
    final_predictions = np.concatenate(predictions)

    # Create a DataFrame with NaN values for the full test period
    full_predictions = pd.DataFrame(index=range(test_length), columns=['prediction'])
    full_predictions['prediction'] = np.nan

    # Fill in the predictions at the correct indices
    for i, pred_window in enumerate(predictions):
        start_idx = i * (window_size + gap)
        end_idx = start_idx + window_size
        full_predictions.iloc[start_idx:end_idx, 0] = pred_window

    return full_predictions

    # Use the function
    pred_RNN4 = windowed_predict(TFT4, train1, covs1, len(test1), window_size=14, gap=30)

    # Transform the predictions and test data
    vt = tn.inverse_transform(test1)
    pt = tn.inverse_transform(pred_TFT4)
    '''
    ###########################################################################################
    '''
    #To do a back-testing follow the below code
    
    tft_models = {
    "TFT_1": TFT4,}
    backtest_models = []

    for model_name, (model_cls, model_kwargs) in tft_models.items():
        model = model_cls(**model_kwargs)
        backtest_models.append(
                model.historical_forecasts(series=train1, future_covariates=covs1,verbose=True)
            )
        mae_val=round(mae(backtest_models[-1], train1), 5)
        std_mae=np.std(np.array(mae_val))
        print(f"{model_name} MAE: {mae_val}± {std_mae:.5f}")
    fix, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, backtest, model_name in zip(
        axes.flatten(),
        backtest_models,
            list(candidates_models.keys()),
        ):
        tn.inverse_transform(train1[-len(backtest) :]).plot(ax=ax, label="ground truth")
        tn.inverse_transform(backtest).plot(ax=ax, label=model_name)

        ax.set_title(model_name)
        ax.set_ylim([0,4])
    plt.tight_layout()
    plt.show()
    '''
    