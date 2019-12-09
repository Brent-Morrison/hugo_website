---
title: Time Series Cross-validation in Python
author: Brent Morrison
date: '2019-12-09'
slug: time-series-cross-validation-in-python
categories:
  - Python
  - Time Series
tags:
  - Python
description: ''
topics: []
---

The content below is a Jupyter Notebook converted to markdown following the instructions of [this](https://www.timlrx.com/2018/03/25/uploading-jupyter-notebook-files-to-blogdown/) blog.   This allows for publishing of Jupyter Notebook's using `blogdown`.  Everything appears to work except for the rendering of tables. That will be something to resolve.  Table formatting notwithstanding, it doesn't really look as nice as the standard Jupyter Notebook format. I've therefore linked the original  [here](https://nbviewer.jupyter.org/github/Brent-Morrison/Misc_scripts/blob/master/TimeSeriesOOS.ipynb).  

## Introduction
This notebook will implement a rolling out of sample forecast on time series data.  The aim of this notebook is to set up the initial infrastructure while allowing for subsequent iteration.  Getting the piping to work, so to speak.  Subsequent iterations will add an inner loop allowing for nested cross validation.  This will be used to tune hyper-parameters in a time series context.  

We will forecast the S&P500 index using an Elastic net logistic regression.  This will be implemented using Scikit Learn's `SGDClassifier` specifying the loss function "log" and penalty "elasticnet".  This specification will result in a regularised logistic regression whereby parameter may be excluded from the model via the shrinkage operator.

A key output of the notebook is a time series of the co-efficients of the  parameters included in model.  This will allow for assessment of the stability of the model over time.  




```
# Import required libraries
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import datetime
import math
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams["figure.figsize"] = (12, 8)
import sklearn.metrics as metrics
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
```

The data.  It comes from [this](https://brentmorrison.netlify.com/post/financial-data-aggregator/) R script.  The data contains an attribute `y1`, which is a binary indicator that looks forward over the next 6 months and returns 1 if the maximum drawdown is more than 20% or the return is less than 2.5%.  

Below we convert dates stored as string to the datetime format and then set this date as the index.


```
# Read csv fle from github
url = 'https://raw.githubusercontent.com/Brent-Morrison/Misc_scripts/master/econ_fin_data.csv'
df_raw = pd.read_csv(url)

# Convert date to datetime and create month end date
df_raw['date'] = pd.to_datetime(df_raw['date'], format='%Y-%m-%d')
df_raw['me_date'] = pd.Index(df_raw['date']).to_period('M').to_timestamp('M')

# Set date as index in new df
df = df_raw.set_index('me_date')

# Inspect csv data - shape
df.shape
```




    (899, 47)



Some simple feature engineering for attributes that may forecast stock returns.  We are not too concerned about the data and model form at this stage. Remember we are just getting the piping to work.


```
# Creation of attributes
df = df.assign(
    CRED_SPRD = df.BAA - df.AAA
    ,YLD_SPRD = df.GS10 - df.FEDFUNDS
    ,LOAN_GROWTH = np.log(df.LOANS / df.LOANS.shift(6))
    )

# Inspect csv data - first and last records
df.iloc[np.r_[0:4, len(df) - 4:len(df)],]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>AAA</th>
      <th>ACDGNO</th>
      <th>AHETPI</th>
      <th>AWHMAN</th>
      <th>BAA</th>
      <th>BOGMBASE</th>
      <th>CFNAIDIFF</th>
      <th>CPIAUCSL</th>
      <th>CPILFESL</th>
      <th>FEDFUNDS</th>
      <th>GS10</th>
      <th>GS2</th>
      <th>INDPRO</th>
      <th>ISRATIO</th>
      <th>KCFSI</th>
      <th>LOANS</th>
      <th>M2SL</th>
      <th>NEWORDER</th>
      <th>PERMIT</th>
      <th>TB3MS</th>
      <th>TWEXMMTH</th>
      <th>UNRATE</th>
      <th>IC4WSA</th>
      <th>NEWORD</th>
      <th>HMI</th>
      <th>P</th>
      <th>D</th>
      <th>E</th>
      <th>CPI</th>
      <th>Fraction</th>
      <th>Rate.GS10</th>
      <th>Price</th>
      <th>Dividend</th>
      <th>Earnings</th>
      <th>CAPE</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>rtn_m</th>
      <th>fwd_rtn_m</th>
      <th>rtn_6m</th>
      <th>min_6m</th>
      <th>dd_6m</th>
      <th>flag</th>
      <th>y1</th>
      <th>diff_flag</th>
      <th>CRED_SPRD</th>
      <th>YLD_SPRD</th>
      <th>LOAN_GROWTH</th>
    </tr>
    <tr>
      <th>me_date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1945-01-31</th>
      <td>1945-01-01</td>
      <td>2.69</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.5</td>
      <td>3.46</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.8372</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.490000</td>
      <td>0.643333</td>
      <td>0.940000</td>
      <td>17.8000</td>
      <td>1945.041667</td>
      <td>2.370</td>
      <td>189.448293</td>
      <td>9.034717</td>
      <td>13.200993</td>
      <td>11.960463</td>
      <td>13.210000</td>
      <td>13.470000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.059795</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.77</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1945-02-28</th>
      <td>1945-02-01</td>
      <td>2.65</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.5</td>
      <td>3.41</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.7818</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.940000</td>
      <td>0.646667</td>
      <td>0.950000</td>
      <td>17.8000</td>
      <td>1945.125000</td>
      <td>2.355</td>
      <td>195.767917</td>
      <td>9.081539</td>
      <td>13.341429</td>
      <td>12.341754</td>
      <td>13.500000</td>
      <td>14.300000</td>
      <td>0</td>
      <td>0.059795</td>
      <td>-0.049455</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.76</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1945-03-31</th>
      <td>1945-03-01</td>
      <td>2.62</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.3</td>
      <td>3.38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.6710</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.930000</td>
      <td>0.650000</td>
      <td>0.960000</td>
      <td>17.8000</td>
      <td>1945.208333</td>
      <td>2.340</td>
      <td>195.627481</td>
      <td>9.128346</td>
      <td>13.481865</td>
      <td>12.323310</td>
      <td>13.390000</td>
      <td>13.610000</td>
      <td>0</td>
      <td>-0.049455</td>
      <td>0.086521</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.76</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1945-04-30</th>
      <td>1945-04-01</td>
      <td>2.61</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>45.0</td>
      <td>3.36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.3664</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.280000</td>
      <td>0.650000</td>
      <td>0.973333</td>
      <td>17.8000</td>
      <td>1945.291667</td>
      <td>2.325</td>
      <td>200.542744</td>
      <td>9.128346</td>
      <td>13.669109</td>
      <td>12.631867</td>
      <td>13.670000</td>
      <td>14.840000</td>
      <td>0</td>
      <td>0.086521</td>
      <td>0.011390</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.75</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-08-31</th>
      <td>2019-08-01</td>
      <td>2.98</td>
      <td>44576.0</td>
      <td>23.60</td>
      <td>41.5</td>
      <td>3.87</td>
      <td>3271378.0</td>
      <td>-0.09</td>
      <td>256.300</td>
      <td>264.245</td>
      <td>2.13</td>
      <td>1.63</td>
      <td>1.57</td>
      <td>109.9273</td>
      <td>1.4</td>
      <td>-0.33</td>
      <td>9883.1084</td>
      <td>14930.9</td>
      <td>68901.0</td>
      <td>1425.0</td>
      <td>1.95</td>
      <td>92.2746</td>
      <td>3.7</td>
      <td>216750.0</td>
      <td>47.2</td>
      <td>67.0</td>
      <td>2897.498182</td>
      <td>56.839092</td>
      <td>134.063333</td>
      <td>256.5580</td>
      <td>2019.625000</td>
      <td>1.630</td>
      <td>2909.712357</td>
      <td>57.078692</td>
      <td>134.628467</td>
      <td>28.704955</td>
      <td>2822.120117</td>
      <td>2926.459961</td>
      <td>79599440000</td>
      <td>-0.018257</td>
      <td>0.017035</td>
      <td>0.049729</td>
      <td>2722.270020</td>
      <td>-0.022599</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.89</td>
      <td>-0.50</td>
      <td>0.022257</td>
    </tr>
    <tr>
      <th>2019-09-30</th>
      <td>2019-09-01</td>
      <td>3.03</td>
      <td>43233.0</td>
      <td>23.67</td>
      <td>41.5</td>
      <td>3.91</td>
      <td>3202682.0</td>
      <td>-0.24</td>
      <td>256.358</td>
      <td>264.595</td>
      <td>2.04</td>
      <td>1.70</td>
      <td>1.65</td>
      <td>109.5940</td>
      <td>1.4</td>
      <td>-0.35</td>
      <td>9902.6579</td>
      <td>15028.3</td>
      <td>68533.0</td>
      <td>1391.0</td>
      <td>1.89</td>
      <td>92.6991</td>
      <td>3.5</td>
      <td>212750.0</td>
      <td>47.3</td>
      <td>68.0</td>
      <td>2982.156000</td>
      <td>57.220000</td>
      <td>133.460000</td>
      <td>256.7590</td>
      <td>2019.708333</td>
      <td>1.700</td>
      <td>2992.382665</td>
      <td>57.416224</td>
      <td>133.917673</td>
      <td>29.228182</td>
      <td>2891.850098</td>
      <td>2976.739990</td>
      <td>73992330000</td>
      <td>0.017035</td>
      <td>0.020226</td>
      <td>0.048998</td>
      <td>2728.810059</td>
      <td>-0.037965</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.88</td>
      <td>-0.34</td>
      <td>0.019858</td>
    </tr>
    <tr>
      <th>2019-10-31</th>
      <td>2019-10-01</td>
      <td>3.01</td>
      <td>42476.0</td>
      <td>23.76</td>
      <td>41.4</td>
      <td>3.92</td>
      <td>3252830.0</td>
      <td>-0.22</td>
      <td>257.271</td>
      <td>265.011</td>
      <td>1.83</td>
      <td>1.71</td>
      <td>1.55</td>
      <td>108.6714</td>
      <td>1.4</td>
      <td>-0.51</td>
      <td>9931.7438</td>
      <td>15195.0</td>
      <td>69267.0</td>
      <td>1461.0</td>
      <td>1.65</td>
      <td>92.3729</td>
      <td>3.6</td>
      <td>215000.0</td>
      <td>49.1</td>
      <td>71.0</td>
      <td>2977.680000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257.3460</td>
      <td>2019.791667</td>
      <td>1.710</td>
      <td>2981.076008</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>28.838509</td>
      <td>2855.939941</td>
      <td>3037.560059</td>
      <td>77564550000</td>
      <td>0.020226</td>
      <td>0.033480</td>
      <td>0.030664</td>
      <td>2728.810059</td>
      <td>-0.076525</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.91</td>
      <td>-0.12</td>
      <td>0.020239</td>
    </tr>
    <tr>
      <th>2019-11-30</th>
      <td>2019-11-01</td>
      <td>3.06</td>
      <td>42476.0</td>
      <td>23.83</td>
      <td>41.4</td>
      <td>3.94</td>
      <td>3315603.0</td>
      <td>-0.22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.55</td>
      <td>1.81</td>
      <td>1.61</td>
      <td>NaN</td>
      <td>1.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.54</td>
      <td>92.4113</td>
      <td>3.5</td>
      <td>217750.0</td>
      <td>47.2</td>
      <td>70.0</td>
      <td>3120.460000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>257.6395</td>
      <td>2019.875000</td>
      <td>1.820</td>
      <td>3120.460000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.008420</td>
      <td>3050.719971</td>
      <td>3140.979980</td>
      <td>72179920000</td>
      <td>0.033480</td>
      <td>0.001568</td>
      <td>0.132185</td>
      <td>2728.810059</td>
      <td>-0.008484</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.88</td>
      <td>0.26</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



To the modelling.  Select the attributes to be predictors and label in our model.


```
# List variables  
# Dependent variable
dep_var = ['y1']
# Predictors/ independent variables
ind_vars = ['CRED_SPRD', 'YLD_SPRD', 'LOAN_GROWTH', 'rtn_6m']
# Other variables to be used in the plot
oth_vars = ['close']
vars = dep_var + ind_vars + oth_vars
df = df[vars]
```

Our new data frame.


```
# Drop na' when variables are not null / Nan
df = df.dropna(subset = vars)

# Inspect csv data - first and last records
df.iloc[np.r_[0:4, len(df) - 4:len(df)],]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y1</th>
      <th>CRED_SPRD</th>
      <th>YLD_SPRD</th>
      <th>LOAN_GROWTH</th>
      <th>rtn_6m</th>
      <th>close</th>
    </tr>
    <tr>
      <th>me_date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1954-07-31</th>
      <td>0.0</td>
      <td>0.61</td>
      <td>1.50</td>
      <td>0.006512</td>
      <td>0.168940</td>
      <td>30.879999</td>
    </tr>
    <tr>
      <th>1954-08-31</th>
      <td>0.0</td>
      <td>0.62</td>
      <td>1.14</td>
      <td>0.016628</td>
      <td>0.131665</td>
      <td>29.830000</td>
    </tr>
    <tr>
      <th>1954-09-30</th>
      <td>0.0</td>
      <td>0.58</td>
      <td>1.32</td>
      <td>0.019759</td>
      <td>0.181765</td>
      <td>32.310001</td>
    </tr>
    <tr>
      <th>1954-10-31</th>
      <td>0.0</td>
      <td>0.59</td>
      <td>1.58</td>
      <td>0.026823</td>
      <td>0.114238</td>
      <td>31.680000</td>
    </tr>
    <tr>
      <th>2019-03-31</th>
      <td>0.0</td>
      <td>1.07</td>
      <td>0.16</td>
      <td>0.029602</td>
      <td>-0.027690</td>
      <td>2834.399902</td>
    </tr>
    <tr>
      <th>2019-04-30</th>
      <td>0.0</td>
      <td>1.01</td>
      <td>0.11</td>
      <td>0.028409</td>
      <td>0.082800</td>
      <td>2945.830078</td>
    </tr>
    <tr>
      <th>2019-05-31</th>
      <td>0.0</td>
      <td>0.96</td>
      <td>0.01</td>
      <td>0.028386</td>
      <td>-0.002943</td>
      <td>2752.060059</td>
    </tr>
    <tr>
      <th>2019-06-30</th>
      <td>0.0</td>
      <td>1.04</td>
      <td>-0.31</td>
      <td>0.025044</td>
      <td>0.159981</td>
      <td>2941.760010</td>
    </tr>
  </tbody>
</table>
</div>



Lets get modelling.  The variables defined in the next cell inform the training and testing ranges.


```
# Set training and testing ranges
train_length = 300
test_length = 4
loops = math.floor((len(df) - train_length) / test_length)
start = len(df) - (loops * test_length + train_length)
stop = math.floor((len(df) - train_length) / test_length) * test_length

# Empty objects
y_pred_prob = None
model_coef = None
```

The training, testing loop.  This piece of code selects a training subset, fits a model and then applies that model to new unseen data.  Predicted probabilities, and parameter co-efficients are returned to a dataframe.  


```
# Training loop
for i in range(start, stop, test_length):

    # Model data
    y_train_raw = np.array(df.iloc[i:i + train_length, 0])
    x_train_raw = np.array(df.iloc[i:i + train_length, 1:len(vars) - 1])
    y_test_raw = np.array(df.iloc[i + train_length:i + train_length + test_length, 0])
    x_test_raw = np.array(df.iloc[i + train_length:i + train_length + test_length, 1:len(vars) - 1])

    # Scale for model ingestion
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train_raw)

    # Apply mean and standard deviation from transform applied to training data to test data
    x_test = sc.transform(x_test_raw)

    # Specify model
    sgd = linear_model.SGDClassifier(
        loss = 'log'
        ,penalty = 'elasticnet'
        ,max_iter = 2500
        ,n_iter_no_change = 500
        ,tol = 1e-3)

    # Train model
    sgd.fit(x_train, y_train_raw)

    # Predict on test data and write to array
    y_pred = sgd.predict_proba(x_test)
    if y_pred_prob is None:
        y_pred_prob = y_pred
    else:
        y_pred_prob = np.concatenate((y_pred_prob, y_pred))

    # Capture co-efficients and write to array
    coef = np.repeat(
        np.append(
            sgd.intercept_[0], 
            sgd.coef_).reshape((1, -1)),
        test_length, 
        axis = 0
        )
    
    if model_coef is None:
        model_coef = coef
    else:
        model_coef = np.concatenate((model_coef, coef))

# Create predictions dataframe with date index
df_preds = pd.DataFrame(
    data = y_pred_prob
    ,index = pd.date_range(
        start = df.index[train_length + start]
        ,periods = stop
        ,freq = 'M')
    ,columns = [0, 1]
    )
# Theshold for hard prediction, to populate confusion matrix
df_preds = df_preds.assign(pred = np.where(df_preds[1] > 0.25, 1, 0))

# Join predictions to df & rename prediction to pred_prob
df_preds = df_preds.join(df, how = 'inner')
df_preds = df_preds.rename(columns = {1:'pred_prob'}).drop(columns = 0)
df_preds.y1 = df_preds.y1.astype(int)
df_preds.close = np.log(df_preds['close'])

# Create co-efficients dataframe with date index
ind_vars.insert(0, 'Int')
ind_vars = [x + '_coef' for x in ind_vars]
df_model_coef = pd.DataFrame(
    data = model_coef
    ,index = pd.date_range(
        start = df.index[train_length + start]
        ,periods = stop
        ,freq = 'M')
    ,columns = ind_vars
    )

# Join predictions & co-efficients df's
df_preds_coefs = df_preds.join(df_model_coef, how = 'inner')
```

Lets inspect the dataframe containing the prediction probability, predictors, intercept and co-efficients.


```
# Inspect dataframe of prediction probability
df_preds_coefs.iloc[np.r_[0:4, len(df_preds_coefs) - 4:len(df_preds_coefs)],]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_prob</th>
      <th>pred</th>
      <th>y1</th>
      <th>CRED_SPRD</th>
      <th>YLD_SPRD</th>
      <th>LOAN_GROWTH</th>
      <th>rtn_6m</th>
      <th>close</th>
      <th>Int_coef</th>
      <th>CRED_SPRD_coef</th>
      <th>YLD_SPRD_coef</th>
      <th>LOAN_GROWTH_coef</th>
      <th>rtn_6m_coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1979-07-31</th>
      <td>0.564997</td>
      <td>1</td>
      <td>0</td>
      <td>1.09</td>
      <td>-1.52</td>
      <td>0.084374</td>
      <td>0.038092</td>
      <td>4.642562</td>
      <td>-0.850876</td>
      <td>-0.059612</td>
      <td>-0.678869</td>
      <td>0.095629</td>
      <td>0.015339</td>
    </tr>
    <tr>
      <th>1979-08-31</th>
      <td>0.611875</td>
      <td>1</td>
      <td>0</td>
      <td>1.12</td>
      <td>-1.91</td>
      <td>0.084071</td>
      <td>0.127019</td>
      <td>4.694279</td>
      <td>-0.850876</td>
      <td>-0.059612</td>
      <td>-0.678869</td>
      <td>0.095629</td>
      <td>0.015339</td>
    </tr>
    <tr>
      <th>1979-09-30</th>
      <td>0.633431</td>
      <td>1</td>
      <td>1</td>
      <td>1.10</td>
      <td>-2.10</td>
      <td>0.085207</td>
      <td>0.073334</td>
      <td>4.694279</td>
      <td>-0.850876</td>
      <td>-0.059612</td>
      <td>-0.678869</td>
      <td>0.095629</td>
      <td>0.015339</td>
    </tr>
    <tr>
      <th>1979-10-31</th>
      <td>0.754928</td>
      <td>1</td>
      <td>0</td>
      <td>1.27</td>
      <td>-3.47</td>
      <td>0.075088</td>
      <td>0.000589</td>
      <td>4.623207</td>
      <td>-0.850876</td>
      <td>-0.059612</td>
      <td>-0.678869</td>
      <td>0.095629</td>
      <td>0.015339</td>
    </tr>
    <tr>
      <th>2019-03-31</th>
      <td>0.436434</td>
      <td>1</td>
      <td>0</td>
      <td>1.07</td>
      <td>0.16</td>
      <td>0.029602</td>
      <td>-0.027690</td>
      <td>7.949586</td>
      <td>-1.462967</td>
      <td>0.089407</td>
      <td>-0.667003</td>
      <td>-0.593875</td>
      <td>-0.700222</td>
    </tr>
    <tr>
      <th>2019-04-30</th>
      <td>0.289830</td>
      <td>1</td>
      <td>0</td>
      <td>1.01</td>
      <td>0.11</td>
      <td>0.028409</td>
      <td>0.082800</td>
      <td>7.988146</td>
      <td>-1.462967</td>
      <td>0.089407</td>
      <td>-0.667003</td>
      <td>-0.593875</td>
      <td>-0.700222</td>
    </tr>
    <tr>
      <th>2019-05-31</th>
      <td>0.421149</td>
      <td>1</td>
      <td>0</td>
      <td>0.96</td>
      <td>0.01</td>
      <td>0.028386</td>
      <td>-0.002943</td>
      <td>7.920105</td>
      <td>-1.462967</td>
      <td>0.089407</td>
      <td>-0.667003</td>
      <td>-0.593875</td>
      <td>-0.700222</td>
    </tr>
    <tr>
      <th>2019-06-30</th>
      <td>0.259595</td>
      <td>1</td>
      <td>0</td>
      <td>1.04</td>
      <td>-0.31</td>
      <td>0.025044</td>
      <td>0.159981</td>
      <td>7.986763</td>
      <td>-1.462967</td>
      <td>0.089407</td>
      <td>-0.667003</td>
      <td>-0.593875</td>
      <td>-0.700222</td>
    </tr>
  </tbody>
</table>
</div>



Now to plotting the prediction probability, predictors, intercept and co-efficients.


```
# Set plot style
sns.set_style('white', {"xtick.major.size": 2, "ytick.major.size": 2})
flatui = ["#c5b4cc", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
sns.set_palette(sns.color_palette(flatui,7))
```


```
# Plot timeseries of SP500, prediction %, and y label shading
fig1, (ax1, ax2) = plt.subplots(nrows = 2)
fig1.suptitle('Prediction probability and S&P 500', size = 16).set_y(1.05)
fig1.subplots_adjust(top = 0.85)

ax1.plot(df_preds_coefs.index, df_preds_coefs['pred_prob'], 'k-', 
    color = sns.xkcd_rgb['grey'])
ax1.fill_between(
    df_preds_coefs.index, 
    df_preds_coefs['pred_prob'], 
    y2 = 0, 
    where = df_preds_coefs['y1']
    )
ax1.set_ylabel('Probability')

ax2.plot(df_preds_coefs.index, df_preds_coefs['close'], 'k-',
    color = sns.xkcd_rgb['grey'])
ax2.fill_between(
    df_preds_coefs.index, 
    df_preds_coefs['close'], 
    y2 = 0, 
    where = df_preds_coefs['y1']
    )
ax2.set_ylim(bottom = 4.5)
ax2.set_ylabel('S&P500 (log scale)')
fig1.tight_layout()
```


![](/TimeSeriesOOS_18_0.png)


Our model is not very good at forecasting the drawdown in the S&P 500.  This is not surprising given the little time put into it.  

Lets now look at the regression parameters.


```
# Plot parameters
fig2, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4)
fig2.suptitle('Rolling regression parameters', size = 16).set_y(1.05)
fig2.subplots_adjust(top = 0.85)

ax1.plot(df_preds_coefs.index, df_preds_coefs['Int_coef'], 'k-', 
    color = sns.xkcd_rgb['grey'])
ax1.fill_between(
    df_preds_coefs.index, 
    df_preds_coefs['Int_coef'], 
    y2 = df_preds_coefs['Int_coef'].min(), 
    where = df_preds_coefs['y1']
    )
ax1.set_ylabel('Intercept')

ax2.plot(df_preds_coefs.index, df_preds_coefs['CRED_SPRD_coef'], 'k-', 
    color = sns.xkcd_rgb['grey'])
ax2.fill_between(
    df_preds_coefs.index, 
    df_preds_coefs['CRED_SPRD_coef'], 
    y2 = df_preds_coefs['CRED_SPRD_coef'].min(),
    where = df_preds_coefs['y1']
    )
ax2.set_ylabel('Credit spread')

ax3.plot(df_preds_coefs.index, df_preds_coefs['YLD_SPRD_coef'], 'k-',
    color = sns.xkcd_rgb['grey'])
ax3.fill_between(
    df_preds_coefs.index, 
    df_preds_coefs['YLD_SPRD_coef'], 
    y2 = df_preds_coefs['YLD_SPRD_coef'].min(),
    where = df_preds_coefs['y1']
    )
ax3.set_ylabel('Yield spread')

ax4.plot(df_preds_coefs.index, df_preds_coefs['LOAN_GROWTH_coef'], 'k-',
    color = sns.xkcd_rgb['grey'])
ax4.fill_between(
    df_preds_coefs.index, 
    df_preds_coefs['LOAN_GROWTH_coef'], 
    y2 = df_preds_coefs['LOAN_GROWTH_coef'].min(),
    where = df_preds_coefs['y1']
    )
ax4.set_ylabel('Loan growth')

fig2.tight_layout()
```


![](/TimeSeriesOOS_20_0.png)


## Conclusion

The code embedded in this notebook is working as expected.  We have produced a rolling out of sample rolling regression, capturing the prediction probability and regression parameters. 

The next steps for this piece of analysis is to embed an inner loop for nested cross validition.  This will enable the tuning model hyper-parameters in a time series context.


## References

https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9 
https://github.com/sam31415/timeseriescv/blob/master/timeseriescv/cross_validation.py  
https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/  
https://www.mikulskibartosz.name/nested-cross-validation-in-time-series-forecasting-using-scikit-learn-and-statsmodels/  
https://arxiv.org/pdf/1905.11744.pdf

