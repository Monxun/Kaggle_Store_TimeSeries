#!/usr/bin/env python
# coding: utf-8

# In[162]:


import sktime
import pandas as pd
import numpy as np
from sktime import datasets


# In[163]:


pd.plotting.deregister_matplotlib_converters()


# In[164]:


airline = datasets.load_airline()


# In[165]:


airline


# In[166]:


from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split


# In[167]:


y_train, y_test, = temporal_train_test_split(airline, test_size=36)


# In[168]:


# y_train, y_test, = temporal_train_test_split(airline, test_size=0.3) ## FOR PERCENT TEST SIZE


# In[169]:


forecaster = NaiveForecaster(strategy="mean")


# In[170]:


fh = list(range(1, 37))


# In[171]:


forecaster.fit(y_train)


# In[172]:


y_pred = forecaster.predict(fh)


# In[173]:


plot_series(y_train, y_test, y_pred, labels = ["train", "test", "pred"])


# In[174]:


forecaster = NaiveForecaster(strategy="drift")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_series(y_train, y_test, y_pred, labels=["Train", "Test", "Pred"])


# In[ ]:





# In[175]:


#TIME SERIES
## UNIVARIATE = one variable at a time for different time periods
## MULTIVARIATE = multiple variables at a time for different time periods


# In[176]:


import matplotlib.pyplot as plt


# In[177]:


from sktime.utils.plotting import plot_series


# In[178]:


plot_series(airline[:30], airline[30:], labels=['First', 'Second'])


# In[179]:


plot_series(y_train, y_test, labels = ["train", "test"])


# In[180]:


forecaster = NaiveForecaster(strategy="last")


# In[181]:


forecaster.fit(y_train)


# In[182]:


y_pred = forecaster.predict(fh)


# In[183]:


plot_series(y_train, y_test, y_pred, labels = ["train", "test", "pred"])

