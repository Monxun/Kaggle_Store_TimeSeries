#!/usr/bin/env python
# coding: utf-8

# In[208]:


import sktime
import pandas as pd
import numpy as np
from sktime import datasets


# In[209]:


pd.plotting.deregister_matplotlib_converters()


# In[210]:


airline = datasets.load_airline()


# In[211]:


airline


# In[212]:


#TIME SERIES
## UNIVARIATE = one variable at a time for different time periods
## MULTIVARIATE = multiple variables at a time for different time periods


# In[213]:


import matplotlib.pyplot as plt


# In[214]:


from sktime.utils.plotting import plot_series


# In[215]:


plot_series(airline[:30], airline[30:], labels=['First', 'Second'])


# In[216]:


from sktime.forecasting.naive import NaiveForecaster


# In[217]:


from sktime.forecasting.model_selection import temporal_train_test_split


# In[218]:


y_train, y_test = temporal_train_test_split(airline, test_size=36)


# In[219]:


len(y_test)


# In[220]:


len(y_train)


# In[221]:


plot_series(y_train, y_test, labels = ["train", "test"])


# In[222]:


forecaster = NaiveForecaster(strategy="last")


# In[223]:


forecaster.fit(y_train)


# In[224]:


fh = np.arange(1, len(y_test) + 1)


# In[225]:


y_pred = forecaster.predict(fh)


# In[226]:


y_pred


# In[227]:


plot_series(y_train, y_test, y_pred, labels = ["train", "test", "pred"])

