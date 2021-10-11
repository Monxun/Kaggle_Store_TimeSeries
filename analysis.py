# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 16:12:22 2021

@author: monxu
"""

import numpy as np
import pandas as pd
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mlxtend.preprocessing import minmax_scaling

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# %% [markdown]
# # Step 2: Load the data
# 
# Next, we'll load the training and test data.  
# 
# We set `index_col=0` in the code cell below to use the `id` column to index the DataFrame.  (*If you're not sure how this works, try temporarily removing `index_col=0` and see how it changes the result.*)

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:48.355918Z","iopub.execute_input":"2021-08-27T21:45:48.356361Z","iopub.status.idle":"2021-08-27T21:45:52.811534Z","shell.execute_reply.started":"2021-08-27T21:45:48.356332Z","shell.execute_reply":"2021-08-27T21:45:52.810525Z"}}
# Load the training data
train = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

# train = train.rename(columns={'sales': 'target'})
# Preview the data
train.head()

# %% [markdown]
# #### Learning more about our data

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:52.813443Z","iopub.execute_input":"2021-08-27T21:45:52.813757Z","iopub.status.idle":"2021-08-27T21:45:53.068779Z","shell.execute_reply.started":"2021-08-27T21:45:52.813727Z","shell.execute_reply":"2021-08-27T21:45:53.067737Z"}}
train.describe()

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:53.070508Z","iopub.execute_input":"2021-08-27T21:45:53.070804Z","iopub.status.idle":"2021-08-27T21:45:53.251818Z","shell.execute_reply.started":"2021-08-27T21:45:53.070777Z","shell.execute_reply":"2021-08-27T21:45:53.250780Z"}}
sns.boxplot(train['sales'])

# %% [markdown]
# ### Observations:
# 
# - Train set has 300,000 rows while test set has 200,000 rows.
# - There are 10 categorical features from `cat0` - `cat9` and 14 continuous features from `cont0` - `cont13`.
# - There is no missing values in the train and test dataset but there is no category `G` in `cat6` test dataset.
# - Categorical features ranging from alphabet A - O but it varies from each categorical feature with `cat0`, `cat1`, `cat3`, `cat5` and `cat6` are dominated by one category.
# - Continuous features on train anda test dataset ranging from -0.1 to 1.25 which are a multimodal distribution and they resemble each other.
# - target has a range between 6.8 to 10.5 and has a bimodal distribution.
# 
# 
# Ideas:
# 
# Drop features that are dominated by one category cat0, cat1, cat3, cat5 and cat6 as they don't give variation to the dataset but further analysis still be needed.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:53.253424Z","iopub.execute_input":"2021-08-27T21:45:53.253804Z","iopub.status.idle":"2021-08-27T21:45:53.259418Z","shell.execute_reply.started":"2021-08-27T21:45:53.253772Z","shell.execute_reply":"2021-08-27T21:45:53.257822Z"}}
cat_features = [feature for feature in train.columns if 'cat' in feature]
cont_features = [feature for feature in train.columns if 'cont' in feature]

# %% [markdown]
# #### Number of rows and columns

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:53.260882Z","iopub.execute_input":"2021-08-27T21:45:53.261243Z","iopub.status.idle":"2021-08-27T21:45:53.273126Z","shell.execute_reply.started":"2021-08-27T21:45:53.261213Z","shell.execute_reply":"2021-08-27T21:45:53.271819Z"}}
print('Rows and Columns in train dataset:', train.shape)
print('Rows and Columns in test dataset:', test.shape)

# %% [markdown]
# #### Number of missing values

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:53.274641Z","iopub.execute_input":"2021-08-27T21:45:53.275019Z","iopub.status.idle":"2021-08-27T21:45:53.786889Z","shell.execute_reply.started":"2021-08-27T21:45:53.274985Z","shell.execute_reply":"2021-08-27T21:45:53.785451Z"}}
print('Missing values in train dataset:', sum(train.isnull().sum()))
print('Missing values in test dataset:', sum(test.isnull().sum()))

# %% [markdown]
# # Step 3: Features and target correlation
# 
# ### Basic statistics on continuous features
# 
# #### Train dataset

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:45:53.790566Z","iopub.execute_input":"2021-08-27T21:45:53.790904Z","iopub.status.idle":"2021-08-27T21:46:16.799454Z","shell.execute_reply.started":"2021-08-27T21:45:53.790873Z","shell.execute_reply":"2021-08-27T21:46:16.798077Z"}}
fig = plt.figure(figsize=(15, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for col in range(0, 4):
    for row in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].set_yticklabels([])
        locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.3, 5.3, 'Continuous Features Distribution on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.3, 4.7, 'Continuous features have multimodal', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for col in cont_features:
    sns.kdeplot(train[col], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(col, fontsize=10, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.2, 1.2)
    locals()["ax"+str(run_no)].set_xlabel('')
    run_no += 1
    
ax14.remove()
ax15.remove()

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:16.801876Z","iopub.execute_input":"2021-08-27T21:46:16.802285Z","iopub.status.idle":"2021-08-27T21:46:17.284809Z","shell.execute_reply.started":"2021-08-27T21:46:16.802249Z","shell.execute_reply":"2021-08-27T21:46:17.283556Z"}}
if cont_features:
    train[cont_features].describe() # to view some basic statistical details of the continuous features

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:17.286401Z","iopub.execute_input":"2021-08-27T21:46:17.286699Z","iopub.status.idle":"2021-08-27T21:46:18.813570Z","shell.execute_reply.started":"2021-08-27T21:46:17.286670Z","shell.execute_reply":"2021-08-27T21:46:18.812389Z"}}
fig = plt.figure(figsize=(10, 3.5), facecolor='#f6f5f5')
gs = fig.add_gridspec(1, 1)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

ax0 = fig.add_subplot(gs[0, 0])
ax0.set_facecolor(background_color)
ax0.set_yticklabels([])
ax0.tick_params(axis='y', which=u'both',length=0)
for s in ["top","right", 'left']:
    ax0.spines[s].set_visible(False)

ax0.text(-0.5, 0.5, 'Target Distribution on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.5, 0.46, 'Target has a bimodal distribution', fontsize=15, fontweight='light', fontfamily='serif')        

sns.kdeplot(train['sales'], ax=ax0, shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
ax0.grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
ax0.set_xlim(-0.5, 10.5)
ax0.set_xlabel('')
ax0.set_ylabel('')

plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:18.815263Z","iopub.execute_input":"2021-08-27T21:46:18.815567Z","iopub.status.idle":"2021-08-27T21:46:18.838498Z","shell.execute_reply.started":"2021-08-27T21:46:18.815539Z","shell.execute_reply":"2021-08-27T21:46:18.837130Z"}}
print('Target')
train['sales'].describe()

# %% [markdown]
# #### Test dataset

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:18.840409Z","iopub.execute_input":"2021-08-27T21:46:18.840727Z","iopub.status.idle":"2021-08-27T21:46:34.623562Z","shell.execute_reply.started":"2021-08-27T21:46:18.840699Z","shell.execute_reply":"2021-08-27T21:46:34.622088Z"}}
fig = plt.figure(figsize=(15, 10), facecolor='#f6f5f5')
gs = fig.add_gridspec(4, 4)
gs.update(wspace=0.2, hspace=0.05)

background_color = "#f6f5f5"

run_no = 0
for col in range(0, 4):
    for row in range(0, 4):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        locals()["ax"+str(run_no)].set_yticklabels([])
        locals()["ax"+str(run_no)].tick_params(axis='y', which=u'both',length=0)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.3, 5.3, 'Continuous Features Distribution on Test Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.3, 4.7, 'Continuous features on test dataset resemble train dataset', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for col in cont_features:
    sns.kdeplot(test[col], ax=locals()["ax"+str(run_no)], shade=True, color='#2f5586', edgecolor='black', linewidth=1.5, alpha=0.9, zorder=3)
    locals()["ax"+str(run_no)].grid(which='major', axis='x', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    locals()["ax"+str(run_no)].set_ylabel(col, fontsize=10, fontweight='bold').set_rotation(0)
    locals()["ax"+str(run_no)].yaxis.set_label_coords(1, 0)
    locals()["ax"+str(run_no)].set_xlim(-0.2, 1.2)
    locals()["ax"+str(run_no)].set_xlabel('')
    run_no += 1
    
ax14.remove()
ax15.remove()

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:34.625741Z","iopub.execute_input":"2021-08-27T21:46:34.626221Z","iopub.status.idle":"2021-08-27T21:46:34.832501Z","shell.execute_reply.started":"2021-08-27T21:46:34.626176Z","shell.execute_reply":"2021-08-27T21:46:34.831215Z"}}
test[cont_features].describe()

# %% [markdown]
# ### Count of categorical features

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:34.834238Z","iopub.execute_input":"2021-08-27T21:46:34.834650Z","iopub.status.idle":"2021-08-27T21:46:37.250442Z","shell.execute_reply.started":"2021-08-27T21:46:34.834610Z","shell.execute_reply":"2021-08-27T21:46:37.249514Z"}}
background_color = "#f6f5f5"

fig = plt.figure(figsize=(25, 8), facecolor=background_color)
gs = fig.add_gridspec(2, 5)
gs.update(wspace=0.2, hspace=0.2)

run_no = 0
for row in range(0, 2):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.8, 115, 'Count of categorical features on Train dataset (%)', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.8, 107, 'Some features are dominated by one category', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for col in cat_features:
    chart_df = pd.DataFrame(train[col].value_counts() / len(train) * 100)
    sns.barplot(x=chart_df.index, y=chart_df[col], ax=locals()["ax"+str(run_no)], color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    run_no += 1

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:37.251817Z","iopub.execute_input":"2021-08-27T21:46:37.252134Z","iopub.status.idle":"2021-08-27T21:46:39.225256Z","shell.execute_reply.started":"2021-08-27T21:46:37.252098Z","shell.execute_reply":"2021-08-27T21:46:39.224052Z"}}
background_color = "#f6f5f5"

fig = plt.figure(figsize=(25, 8), facecolor=background_color)
gs = fig.add_gridspec(2, 5)
gs.update(wspace=0.2, hspace=0.2)

run_no = 0
for row in range(0, 2):
    for col in range(0, 5):
        locals()["ax"+str(run_no)] = fig.add_subplot(gs[row, col])
        locals()["ax"+str(run_no)].set_facecolor(background_color)
        for s in ["top","right", 'left']:
            locals()["ax"+str(run_no)].spines[s].set_visible(False)
        run_no += 1

ax0.text(-0.8, 109, 'Count of categorical features on Test dataset (%)', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(-0.8, 101, 'Some features are dominated by one category', fontsize=13, fontweight='light', fontfamily='serif')        

run_no = 0
for col in cat_features:
    chart_df = pd.DataFrame(test[col].value_counts() / len(test) * 100)
    sns.barplot(x=chart_df.index, y=chart_df[col], ax=locals()["ax"+str(run_no)], color='#2f5586', zorder=3, edgecolor='black', linewidth=1.5)
    locals()["ax"+str(run_no)].grid(which='major', axis='y', zorder=0, color='gray', linestyle=':', dashes=(1,5))
    run_no += 1

# %% [markdown]
# ### Features Correlation
# 
# Observations:
# 
# - Highest correlation between features is 0.5.
# - Correlation between features on train and test dataset are quite similar.

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:39.226509Z","iopub.execute_input":"2021-08-27T21:46:39.226788Z","iopub.status.idle":"2021-08-27T21:46:41.676022Z","shell.execute_reply.started":"2021-08-27T21:46:39.226763Z","shell.execute_reply":"2021-08-27T21:46:41.674740Z"}}
background_color = "#f6f5f5"

fig = plt.figure(figsize=(18, 8), facecolor=background_color)
gs = fig.add_gridspec(1, 2)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
colors = ["#2f5586", "#f6f5f5","#2f5586"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

ax0.set_facecolor(background_color)
ax0.text(0, -1, 'Features Correlation on Train Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax0.text(0, -0.4, 'Highest correlation in the dataset is 0.6', fontsize=13, fontweight='light', fontfamily='serif')

ax1.set_facecolor(background_color)
ax1.text(-0.1, -1, 'Features Correlation on Test Dataset', fontsize=20, fontweight='bold', fontfamily='serif')
ax1.text(-0.1, -0.4, 'Features in test dataset resemble features in train dataset ', 
         fontsize=13, fontweight='light', fontfamily='serif')

sns.heatmap(train[cont_features].corr(), ax=ax0, vmin=-1, vmax=1, annot=True, square=True, 
            cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1g')

sns.heatmap(test[cont_features].corr(), ax=ax1, vmin=-1, vmax=1, annot=True, square=True, 
            cbar_kws={"orientation": "horizontal"}, cbar=False, cmap=colormap, fmt='.1g')

plt.show()

# %% [markdown]
# ### Feature Engineering
# 
# Feature-engineering using histograms of the cont features show multiple components. For instance, the `cont1` has 7 discrete peaks as shown below. 

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:41.677286Z","iopub.execute_input":"2021-08-27T21:46:41.677565Z","iopub.status.idle":"2021-08-27T21:46:47.289590Z","shell.execute_reply.started":"2021-08-27T21:46:41.677539Z","shell.execute_reply":"2021-08-27T21:46:47.288443Z"}}
all_data = pd.concat([train, test])

fig, ax = plt.subplots(5, 3, figsize=(14, 24))
for i, feature in enumerate(cont_features):
    plt.subplot(5, 3, i+1)
    sns.histplot(all_data[feature][::100], 
                 color="blue", 
                 kde=True, 
                 bins=100)
    plt.xlabel(feature, fontsize=9)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:47.290846Z","iopub.execute_input":"2021-08-27T21:46:47.291149Z","iopub.status.idle":"2021-08-27T21:46:49.492059Z","shell.execute_reply.started":"2021-08-27T21:46:47.291123Z","shell.execute_reply":"2021-08-27T21:46:49.490920Z"}}
fig, ax = plt.subplots(5, 3, figsize=(24, 30))
for i, feature in enumerate(cont_features):
    plt.subplot(5, 3, i+1)
    sns.scatterplot(x=feature, 
                    y="target", 
                    data=train[::150], 
                    palette='muted')
    plt.xlabel(feature, fontsize=9)
plt.show()

# %% [markdown]
# # Step 4: Prepare the data
# 
# The next code cell separates the target (which we assign to `y`) from the training features (which we assign to `features`).

# %% [code] {"execution":{"iopub.status.busy":"2021-08-27T21:46:49.493443Z","iopub.execute_input":"2021-08-27T21:46:49.493762Z","iopub.status.idle":"2021-08-27T21:46:49.570788Z","shell.execute_reply.started":"2021-08-27T21:46:49.493732Z","shell.execute_reply":"2021-08-27T21:46:49.570104Z"}}
# Separate target from features
y = train['sales']
features = train.drop(['sales'], axis=1)

# List of features for later use
feature_list = list(features.columns)

# Preview features
features.head()