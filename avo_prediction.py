import pandas as pd
import numpy as np
import preprocess_avo as prep
import univariate_models as um
import multivariate_models as mm
import RNN_forecastmodel as rnn_fore
import TFT_forecastmodel as tft_fore
'''
#Use the below code if your using your own data
#avo_data=prep.dataprep(pd.read_csv(""))
#weather_data=prep.weather_VI_dataprep(pd.read_csv(""))
#vegetation_data=prep.weather_VI_dataprep(pd.read_csv(""))
#dataset1=avo_data.merge(weather_data,how='left',on='ds').merge(vegetation_data,how='left',on='ds') 
#fillna is alread handled, so the below is not necessery
#dataset1 = dataset1.fillna(dataset1.mean())
#dataset1=dataset1.set_index('ds')
#check_and_visualize_normality(dataset1)
#dataset1=dataset1.reset_index()
# I am giving synthetic data in this repo, please use if required to use global model You need multiple dataset
#here I am using same but please use different data
'''
avo_data1=pd.read_csv("synthetic_data1.csv")
avo_data2=pd.read_csv("synthetic_data2.csv")
dataset1=avo_data1.copy()
dataset2=avo_data2.copy()
TS_data1,covs1,train1,test1,yield_scale,cov_scale=prep.DL_forecast(dataset1)
TS_data2,covs2,train2,test2,yield_scale,cov_scale=prep.DL_forecast(dataset2)
um.univariate_model(train1,yield_scale)
mm.multivariate_model(train1,yield_scale)
rnn_fore.rnn_forecast(train1,train2,test1,test2,yield_scale,covs1,covs2)
tft_fore.tft_forecast(train1,train2,test1,test2,yield_scale,covs1,covs2)