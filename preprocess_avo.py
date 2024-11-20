import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae,r2_score,rmse,smape,mape
from darts.models import TFTModel
from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.utils.likelihood_models import QuantileRegression
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
scaler = MinMaxScaler()
torch.manual_seed(1)
np.random.seed(1)
scaler = MinMaxScaler()

tn = Scaler(scaler) #scale the main yield time series
sn=Scaler(scaler)#scale future covariate

def smooth_data(df,x):
    x_filtered = df[[x]].apply(savgol_filter,  window_length=31, polyorder=2)
    return x_filtered

# For Yield data
def data_prep(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['y']=(data['Sum_of_TE']*5.5)/data['Hectares']/1000
    object_columns = data.select_dtypes(include=['object']).columns
    for col in object_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # Create the date range (optional, you could change or not to use it )
    date_range = pd.date_range(start='2026-01-01', end='2031-06-19', freq='D')

    # Create the DataFrame with new date range The new date range and datframe is to align all the data set in time period(t-T)
    dummy_df = pd.DataFrame({'Date': date_range})

    # Assuming your actual DataFrame is called 'df' and has a 'Date' column
    # If your date column has a different name, replace 'Date' with the actual column name
    df= dummy_df.merge(data, on='Date', how='left')

    # Sort the DataFrame by date
    df = df.sort_values('Date')
    df=df.set_index('Date')
    result = df.resample('D').asfreq()
    ffill_columns = ['region', 'Hectares', 'IBI']
    result[ffill_columns] = result[ffill_columns].ffill()
    result[ffill_columns] = result[ffill_columns].bfill()
    zero_fill_columns = ['Sum_of_TE', 'Average_fruit_size', 'y']
    result[zero_fill_columns] = result[zero_fill_columns].fillna(0)
    result=result.reset_index()
    result = result.fillna(0)
    result=result.rename(columns={'Date': 'ds'})
    def cumulative_yield(data):
    #Define season
        def map_season(date):
            year = date.year
            if date.month >= 8:
                if date.month <= 12:
                    return f"{year}1"
            elif date.month <= 3:
                return f"{year-1}1"
            elif 4 <= date.month <= 7:
                return f"{year}0"
            return None
        data['Season'] = data['ds'].apply(map_season)
        #get cumulative yield
        def is_avo_season(ds):
            date = pd.to_datetime(ds)
            return (date.month >8 & date.month <3)
        data['on_season'] = data['ds'].apply(is_avo_season)
        data['off_season'] = ~data['ds'].apply(is_avo_season)
        data.replace({False: 0, True: 1}, inplace=True)
        data['y']=(data['y']*data['off_season']).groupby(data['Season']).cumsum()
        data.drop([ 'region','Season'], axis=1, inplace=True)
    result=cumulative_yield(result)
    return result

def weather_VI_dataprep(data):
    data['Date'] = pd.to_datetime(data['Date'],errors='coerce')
    data=data.set_index('Date')
    full_date_rng = pd.date_range(start='2026-01-01', end='2031-06-19', freq='D')
    new_df = pd.DataFrame(index=full_date_rng)
    new_df = new_df.join(data)
    #weather and vegetation index uses monthly mean to fill the gaps for each missing value in the corresponding month
    monthly_mean = new_df.resample('M').mean()
    if new_df.isnull().values.any():
        for column in new_df.columns:
            new_df[column] = new_df[column].fillna(new_df[column].resample('M').transform('mean'))
    new_df = new_df.fillna(method='ffill')
    new_df=new_df.reset_index()
    new_df=new_df.rename(columns={'index': 'ds'})
    return new_df
#check Normality check and Perform Shapiro-Wilk test
def check_and_visualize_normality(df):
    # Set up the plot
    n_cols = len(df.columns)
    fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5*n_cols))
    #fig.suptitle('Normality Check for Each Column', fontsize=16)

    for i, column in enumerate(df.columns):
        data = df[column].dropna()
        
        # Perform Shapiro-Wilk test
        stat, p = stats.shapiro(data)
        
        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[i, 0])
        axes[i, 0].set_title(f'Q-Q Plot: {column}')
        
        # Histogram with KDE
        sns.histplot(data, kde=True, ax=axes[i, 1])
        axes[i, 1].set_title(f'Histogram: {column}')
        
        # Add test results to the histogram plot
        axes[i, 1].text(0.05, 0.95, f'Shapiro-Wilk test:\nStatistic={stat:.3f}, p-value={p:.3f}', 
                        transform=axes[i, 1].transAxes, verticalalignment='top')
        
        if p > 0.05:
            axes[i, 1].text(0.05, 0.85, 'Likely Gaussian', transform=axes[i, 1].transAxes, 
                            verticalalignment='top', color='green')
        else:
            axes[i, 1].text(0.05, 0.85, 'Likely Non-Gaussian', transform=axes[i, 1].transAxes, 
                            verticalalignment='top', color='red')

    plt.tight_layout()
    plt.show()


def DL_forecast(d1):
    cols=['Sum_of_TE', 'Average_fruit_size', 'Hectares', 'IBI',
          'NDVI', 'EVI', 'SAVI',
          'WDRVI', 'GNDVI', 'NDII', 'SMI', 'WSI', 'NDMI', 'Wdir', 'WSpd',
          'GustDir', 'GustSpd', 'WindRun', 'Rain_avg', 'Tdry', 'Twet', 'RH',
          'Tmax', 'Tmin', 'Tgmin', 'Pmsl', 'Pstn', 'Rad', 'Rain_daily',
          'Rain_Deficit', 'Runoff', 'on_season', 'off_season']

    ts=TimeSeries.from_dataframe(d1, 'ds', 'y') #yield time series
    cov=TimeSeries.from_dataframe(d1, 'ds', cols) #covariate columns
    # define percentiles
    figsize = (20, 6)
    lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
    label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
    label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"
    TS_data=tn.fit_transform(ts)
    covs = sn.fit_transform(cov) #scale all covariates
    train,test=TS_data.split_after(pd.Timestamp("2030-12-31 00:00:00"))
    return TS_data,covs,train,test,tn,sn
