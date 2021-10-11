#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# I just started exploring sktime and figured this competition would be a good learning opportunity. 
# Obviously there is A TON you could do but this is some good boilerplate.

# Feel free to reach out to work together =)

# Everything in this notebook was adapted from this awesome article: 
# https://towardsdatascience.com/a-lightgbm-autoregressor-using-sktime-6402726e0e7b


# In[ ]:


pip install sktime


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import itertools

import matplotlib.pyplot as plt
import lightgbm as lgb
from pylab import rcParams
from sklearn.metrics import mean_squared_log_error
from statsmodels.tsa.seasonal import seasonal_decompose


rcParams['figure.figsize'] = 18, 8
rcParams['figure.figsize'] = 18, 8


# In[ ]:


from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.model_selection import (
    temporal_train_test_split,
)
from sktime.utils.plotting import plot_series
from sktime.forecasting.compose import (
    TransformedTargetForecaster,
    make_reduction
)


# In[ ]:


# W/ DATE COLUMN - 'parse_dates='date' (date column)
train = pd.read_csv('../input/store-sales-time-series-forecasting/train.csv', parse_dates=['date'])
transactions = pd.read_csv('../input/store-sales-time-series-forecasting/transactions.csv', parse_dates=['date'])
oil = pd.read_csv('../input/store-sales-time-series-forecasting/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('../input/store-sales-time-series-forecasting/holidays_events.csv', parse_dates=['date'])
test = pd.read_csv('../input/store-sales-time-series-forecasting/test.csv', parse_dates=['date'])


# In[ ]:


# NO DATE COLUMN - 'index_col=0' (index/id column)
stores = pd.read_csv('../input/store-sales-time-series-forecasting/stores.csv', index_col=0)
sample = pd.read_csv('../input/store-sales-time-series-forecasting/sample_submission.csv', index_col=0)


# In[ ]:


train.head()


# In[ ]:


#For the sake of demonstration, we will train our model on monthly aggregated Sales data of a particular store# Select sales for Store 1 Only.
store1_agg = train.loc[train['store_nbr']==1].groupby(['date'])['sales'].sum()
store1_agg.index = pd.to_datetime(store1_agg.index)
#Aggregate the Data on a Monthly basis.
store1_agg_monthly = store1_agg.resample('M').sum()


# In[ ]:


#--------------------Visulaize Data on a Time Plot------------------
sns.lineplot(
    data=store1_agg_monthly, 
)
plt.title("Store-1 Sales Data aggreagted at Month Level")
plt.show()


# In[ ]:


#Annual Seasonal Decomposition
seasonal_decompose(store1_agg_monthly,model="multiplicative",period=12).plot()
plt.show()


# In[ ]:


#--------------------Time Series Train-Test split-------------------#
store1_agg_monthly.index = store1_agg_monthly.index.to_period('M') 
y_train, y_test = temporal_train_test_split(store1_agg_monthly, test_size=0.2)


# In[ ]:


y_train.head()


# In[ ]:


y_train.tail()


# In[ ]:


#--------------------------Detrender-----------------------------

#degree=1 for Linear
forecaster = PolynomialTrendForecaster(degree=1) 
transformer = Detrender(forecaster=forecaster)

#Get the residuals after fitting a linear trend
y_resid = transformer.fit_transform(y_train)

# Internally, the Detrender uses the in-sample predictions
# of the PolynomialTrendForecaster
forecaster = PolynomialTrendForecaster(degree=1)
fh_ins = -np.arange(len(y_train))  # in-sample forecasting horizon
y_pred = forecaster.fit(y_train).predict(fh=fh_ins)
plot_series(y_train, y_pred, y_resid, labels=["y_train", "fitted linear trend", "residuals"]);


# In[ ]:


#--------------------------Deseasonalizer---------------------------

#Multiplicative Deseasonalizer, period = 12(for Monthly Data)
deseasonalizer = Deseasonalizer(model="multiplicative", sp=12)
plot_series(deseasonalizer.fit_transform(y_train))
seasonal = deseasonalizer.fit_transform(y_train)


# In[ ]:


regressor = lgb.LGBMRegressor()


# In[ ]:


forecaster = make_reduction(
                    #hyper-paramter to set recursive strategy
                    estimator=regressor, window_length=4,strategy="recursive" 
)


# In[ ]:


#----------------------------Create Pipeline--------------------

def get_transformed_target_forecaster(alpha,params):
    
    #Initialize Light GBM Regressor   
    regressor = lgb.LGBMRegressor(alpha = alpha,**params)
    
#-----------------------Forecaster Pipeline-----------------
    
    #1.Separate the Seasonal Component.
    #2.Fit a forecaster for the trend.
    #3.Fit a Autoregressor to the resdiual(autoregressing on four historic values).
    
    forecaster = TransformedTargetForecaster(
        [
            ("deseasonalise", Deseasonalizer(model="multiplicative", sp=12)),
            ("detrend", Detrender(forecaster=PolynomialTrendForecaster(degree=1))),
            (
                # Recursive strategy for Multi-Step Ahead Forecast.
                # Auto Regress on four previous values
                "forecast",
                make_reduction(
                    estimator=regressor, window_length=4, strategy="recursive",
                ),
            ),
        ]
    )
    return forecaster


# In[ ]:


#-------------------Fitting an Auto Regressive Light-GBM------------

#Setting Quantile Regression Hyper-parameter.
params = {
    'objective':'quantile'
}
#A 10 percent and 90 percent prediction interval(0.1,0.9 respectively).
quantiles = [.1, .5, .9] #Hyper-parameter "alpha" in Light GBM#Capture forecasts for 10th/median/90th quantile, respectively.
forecasts = []#Iterate for each quantile.
for alpha in quantiles:
    
    forecaster = get_transformed_target_forecaster(alpha,params)
    
    #Initialize ForecastingHorizon class to specify the horizon of forecast
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    
    #Fit on Training data.
    forecaster.fit(y_train)
    
    #Forecast the values.
    y_pred = forecaster.predict(fh)
    
    #List of forecasts made for each quantile.
    y_pred.index.name="date"
    y_pred.name=f"predicted_sales_q_{alpha}"
    forecasts.append(y_pred)
    
#Append the actual data for plotting.
store1_agg_monthly.index.name = "date"
store1_agg_monthly.name = "original"
forecasts.append(store1_agg_monthly)


# In[ ]:


error = mean_squared_log_error(y_test, y_pred)


# In[ ]:


#-------------------Final Plotting of Forecasts------------------

plot_data = pd.melt(pd.concat(forecasts,axis=1).reset_index(), id_vars=['date'],        value_vars=['predicted_sales_q_0.1', 'predicted_sales_q_0.5',
                   'predicted_sales_q_0.9','original'])
plot_data['date'] = pd.to_datetime(plot_data['date'].astype(str).to_numpy())
plot_data['if_original'] = plot_data['variable'].apply(
    lambda r:'original' if r=='original' else 'predicted' 
)
sns.lineplot(data = plot_data,
        x='date',
        y='value',
        hue='if_original',
             style="if_original",
        markers=['o','o'],
)

plt.title(f"Final Forecast - Error: {error}")
plt.show()


# References
# 
#     Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. ( 2013). Machine Learning Strategies for Time Series Forecasting — https://www.researchgate.net/publication/236941795_Machine_Learning_Strategies_for_Time_Series_Forecasting.
#     Markus Löning, Anthony Bagnall, Sajaysurya Ganesh, Viktor Kazakov, Jason Lines, Franz Király (2019): “sktime: A Unified Interface for Machine Learning with Time Series”
#     LightGBM-Quantile loss — https://towardsdatascience.com/lightgbm-for-quantile-regression-4288d0bb23fd
