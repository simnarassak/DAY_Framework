# Day Framework
The DAY framework employs machine-learning techniques to integrate weather data, vegetation indices, historical yield information, and management practices for forecasting daily avocado yield volume. The framework consists of six key processes, data-data acquisition, data pre-processing, model training, model validation and comparison, selected model testing and feature influence and yield variance. Figure \ref{fig:ML_Framework} illustrates the architecture of DAY Framework.

I conducted a comprehensive comparison of various univariate and multivariate models for time series forecasting. The univariate models evaluated include Linear Regression, Exponential Smoothing, Kalman Forecaster, Random Forest, XGBoost, and Temporal Fusion Transformer (TFT). For the multivariate analysis, I examined Multivariate Regression, Bayesian Ridge Regression, Random Forest, XGBoost, Vanilla Recurrent Neural Network (Vanilla-RNN), Long Short-Term Memory Recurrent Neural Network (LSTM-RNN), Gated Recurrent Unit Recurrent Neural Network (GRU-RNN),TFT.  Additionally, I also included RNN and TFT model’s global modeling characteristics. 

The below figure Forecasting performed using GRU-RNN

<img src="image\RNN_Forecastplot.png" width="\linewidth"/>

The below figure Forecasting performed using TFT
<img src="image\TFT_Forecastplot.png" width="\linewidth"/>
