#!/usr/bin/env python
# coding: utf-8

# In[87]:


import sktime
import pandas as pd
from sktime import datasets


# In[88]:


pd.plotting.deregister_matplotlib_converters()


# In[89]:


airline = datasets.load_airline()


# In[90]:


airline


# In[ ]:





# In[91]:


#TIME SERIES
## UNIVARIATE = one variable at a time for different time periods
## MULTIVARIATE = multiple variables at a time for different time periods


# In[92]:


import matplotlib.pyplot as plt


# In[93]:


from sktime.utils.plotting import plot_series


# In[94]:


plot_series(airline)

