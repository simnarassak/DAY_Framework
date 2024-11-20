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
from darts.models import RNNModel
from darts.utils.callbacks import TFMProgressBar
from darts.utils.likelihood_models import GaussianLikelihood


def rnn_forecast(train1,train2,test1,test2,tn,covs1,covs2):
    
    '''
    To use Vanila RNN select model="RNN", for LSTM- model="LSTM" and for GRU- model="GRU"
    '''
    RNN_forecast_model = RNNModel(
    model="GRU",
    hidden_dim=20,
    n_rnn_layers=2,
    dropout=0.1,
    batch_size=30,
    n_epochs=100,
    optimizer_kwargs={"lr": 1e-3},
    random_state=0,
    training_length=50,
    input_chunk_length=30,
    add_encoders={
        "cyclic": {"future": ["minute", "hour", "dayofweek", "month"]},
    'transformer':Scaler(scaler)},
    likelihood=GaussianLikelihood(),
    save_checkpoints=True,  # store the latest and best performing epochs
    force_reset=True,)
    
    #SOM model
    #RNN_GRU1=RNN_forecast_model.fit(series=train1,verbose=True)
    #SOMEV model
    #RNN_GRU2=RNN_forecast_model.fit(series=train1,future_covariates=covs1,verbose=True)
    #MOM model
    #RNN_GRU3=RNN_forecast_model.fit(series=[train1,train2],verbose=True)
    
    #MOMEV model
    RNN_GRU4=RNN_forecast_model.fit(series=[train1,train2],future_covariates=[covs1,covs2],verbose=True)
    
    # prediction
    pred_RNN4 = RNN_GRU4.predict(n=14, series=train1,future_covariates=covs1,verbose=True)
    
    
    pr1=tn.inverse_transform(pred_RNN4)
    df_rnn1=pr1.pd_dataframe()
    df_rnn1=df_rnn1.reset_index()
    df_rnn1=df_rnn1.rename(columns={'y': 'MOMEV'})
    
    tdf=tn.inverse_transform(test1)
    test_df=tdf.pd_dataframe()
    test_df=test_df.reset_index()
    test_df=test_df.rename(columns={'y': 'Observed Yield'})
    
    RNN_prediction_df=test_df.merge(df_rnn1,how="left",on="ds")
    # plot
    plt.figure(figsize=(12, 8))

    # Plot the original data
    plt.plot(RNN_prediction_df.ds,RNN_prediction_df['Observed Yield'],label='Observed Yield',color='black')

    # Plot model predictions
    plt.plot(RNN_prediction_df.ds,RNN_prediction_df.MOMEV, label='MOMEV', linestyle='--')

    plt.legend()

    plt.grid(True, alpha=0.5)
    plt.figure(figsize=(12, 4))

    # Calculate differences
    diff4 = RNN_prediction_df['Observed Yield']- RNN_prediction_df.MOMEV

    # Plot differences
    plt.plot(RNN_prediction_df.ds, diff4, label='MOMEV', color='orange')

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
    pred_RNN4 = windowed_predict(RNN_GRU4, train1, covs1, len(test1), window_size=14, gap=30)

    # Transform the predictions and test data
    vt = tn.inverse_transform(test1)
    pt = tn.inverse_transform(pred_RNN4)
    '''
    #######################################################################################
    '''
    #To do a back-testing follow the below code
    rnn_models = {
    "RNN_1": RNN_GRU4,}
    backtest_models = []

    for model_name, (model_cls, model_kwargs) in rnn_models.items():
        model = model_cls(**model_kwargs)
        backtest_models.append(
                model.historical_forecasts(series=df_train, future_covariates=covs,verbose=True)
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
        tn.inverse_transform(df_train[-len(backtest) :]).plot(ax=ax, label="ground truth")
        tn.inverse_transform(backtest).plot(ax=ax, label=model_name)

        ax.set_title(model_name)
        ax.set_ylim([0,4])
    plt.tight_layout()
    plt.show()
    '''
