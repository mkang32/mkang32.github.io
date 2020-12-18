---
layout: post
title:  "How To Use FB Prophet for Time-series Forecasting: Vehicle Traffic Volume"
date:   2020-12-15 00:00:00 -0400
categories: python
---

Recently, I came across a few articles mentioning Facebook's Prophet library that looked interesting (although the 
initial release was almost 3 years ago!), so I decided to dig more into it.

Prophet is an open-source library developed by Facebook which aims to make time-series forecasting easier and more 
scalable.
 It is a type of generalized additive model (GAM), which uses a regression model with potentially non-linear 
 smoothers. It is called additive because it adds multiple decomposed parts to explain some trends. For example, 
 Prophet uses the following components: 

$$ y(t) = g(t) + s(t) + h(t) + e(t) $$

where,  
$g(t)$: Growth. Big trend. Non-periodic changes.   
$s(t)$: Seasonality. Periodic changes (e.g. weekly, yearly, etc.) represented by Fourier Series.  
$h(t)$: Holiday effect that represents irregular schedules.   
$e(t)$: Error. Any idiosyncratic changes not explained by the model.  

In this post, I will explore main concepts and API endpoints of the Prophet library.

# Table of Contents 
1. [Prepare Data](#prep)
2. [Train And Predict](#train)
3. [Check Components](#components)
4. [Evaluate](#eval)
5. [Trend Change Points](#trend)
6. [Seasonality Mode](#season)
7. [Saving Model](#save)
8. [References](#ref)

<a id=prep></a>
# 1. Prepare Data

In this post. We will use the U.S. traffic volume data available 
[here](https://fred.stlouisfed.org/series/TRFVOLUSM227NFWA), which is a monthly traffic volume (miles traveled) on 
public roadways from January 1970 until September 2020. The unit is a million miles. 


```python
import pandas as pd
import matplotlib.pyplot as plt

# to mute Pandas warnings Prophet needs to fix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```


```python
df.head()
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
      <th>DATE</th>
      <th>TRFVOLUSM227NFWA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01</td>
      <td>80173.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-02-01</td>
      <td>77442.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-03-01</td>
      <td>90223.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-04-01</td>
      <td>89956.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1970-05-01</td>
      <td>97972.0</td>
    </tr>
  </tbody>
</table>
</div>



Prophet is hard-coded to use specific column names; `ds` for dates and `y` for the target variable we want to predict.


```python
# Prophet requires column names to be 'ds' and 'y' 
df.columns = ['ds', 'y']
# 'ds' needs to be datetime object
df['ds'] = pd.to_datetime(df['ds'])
```

When plotting the original data, we can see there is a **big, growing trend** in the traffic volume, although there
seems to be some stagnant or even decreasing trends (**change of rate**) around 1980, 2008, and most strikingly, 2020
. Checking how Prophet can handle these changes would be interesting. There is also a **seasonal, periodic trend** that seems to repeat each year. It goes up until the middle of the year and goes down again. Will Prophet capture this as well?

<div style="text-align:center">
<img src="{{site.baseurl}}/images/2020-12-15-prophet-intro/output_7_0.png">
</div>
<br>
    


For train test split, do not forget that we cannot do a random split for time-series data. We use ONLY the earlier 
part of data for training and the later part of data for testing given a cut-off point. Here, we use 2019/1/1 as our
 cut-off point. 


```python
# split data 
train = df[df['ds'] < pd.Timestamp('2019-01-01')]
test = df[df['ds'] >= pd.Timestamp('2019-01-01')]
```


```python
print(f"Number of months in train data: {len(train)}")
print(f"Number of months in test data: {len(test)}")
```

<pre class="output">
    Number of months in train data: 588
    Number of months in test data: 21
</pre>


<a id=train></a>
# 2. Train And Predict

Let's train a Prophet model. You just initialize an object and `fit`! That's all.

Prophet warns that it disabled weekly and daily seasonality. That's fine because our data set is monthly so there is no weekly or daily seasonality.


```python
from fbprophet import Prophet 

# fit model - ignore train/test split for now 
m = Prophet()
m.fit(train)
```

<pre class="output">
    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.

    <fbprophet.forecaster.Prophet at 0x121b8dc88>
</pre>


When making predictions with Prophet, we need to prepare a special object called future dataframe. It is a Pandas DataFrame with a single column `ds` that includes all datetime within the training data plus additional periods given by user. 

The parameter `periods` is basically the number of points (rows) to predict after the end of the training data. The 
interval (parameter `freq`) is set to 'D' (day) by default, so we need to adjust it to 'MS' (month start) as our data
 is monthly. I set `periods=21` as it is the number of points in the test data.


```python
# future dataframe - placeholder object
future = m.make_future_dataframe(periods=21, freq='MS') 
```


```python
# start of the future df is same as the original data 
future.head()
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
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-02-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-03-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-04-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1970-05-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
# end of the future df is original + 21 periods (21 months)
future.tail()
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
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>604</th>
      <td>2020-05-01</td>
    </tr>
    <tr>
      <th>605</th>
      <td>2020-06-01</td>
    </tr>
    <tr>
      <th>606</th>
      <td>2020-07-01</td>
    </tr>
    <tr>
      <th>607</th>
      <td>2020-08-01</td>
    </tr>
    <tr>
      <th>608</th>
      <td>2020-09-01</td>
    </tr>
  </tbody>
</table>
</div>



It's time to make actual predictions. It's simple - just `predict` with the placeholder DataFrame `future`. 


```python
# predict the future
forecast = m.predict(future)
```

Prophet has a nice built-in plotting function to visualize forecast data. Black dots are for actual data and the blue 
line is prediction. You can also use matplotlib functions to adjust the figure, such as adding legend or adding 
xlim or ylim.


```python
# Prophet's own plotting tool to see 
fig = m.plot(forecast)
plt.legend(['Actual', 'Prediction', 'Uncertainty interval'])
plt.show()
```


<div style="text-align:center">
<img src="{{site.baseurl}}/images/2020-12-15-prophet-intro/output_20_0.png">
</div>
<br>
    


<a id=components></a>
# 3. Check Components

So, what is in the forecast DataFrame? Let's take a look.


```python
forecast.head()
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
      <th>ds</th>
      <th>trend</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>trend_lower</th>
      <th>trend_upper</th>
      <th>additive_terms</th>
      <th>additive_terms_lower</th>
      <th>additive_terms_upper</th>
      <th>yearly</th>
      <th>yearly_lower</th>
      <th>yearly_upper</th>
      <th>multiplicative_terms</th>
      <th>multiplicative_terms_lower</th>
      <th>multiplicative_terms_upper</th>
      <th>yhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01</td>
      <td>94281.848744</td>
      <td>69838.269924</td>
      <td>81366.107613</td>
      <td>94281.848744</td>
      <td>94281.848744</td>
      <td>-18700.514310</td>
      <td>-18700.514310</td>
      <td>-18700.514310</td>
      <td>-18700.514310</td>
      <td>-18700.514310</td>
      <td>-18700.514310</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75581.334434</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-02-01</td>
      <td>94590.609819</td>
      <td>61661.016554</td>
      <td>73066.758942</td>
      <td>94590.609819</td>
      <td>94590.609819</td>
      <td>-27382.307301</td>
      <td>-27382.307301</td>
      <td>-27382.307301</td>
      <td>-27382.307301</td>
      <td>-27382.307301</td>
      <td>-27382.307301</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>67208.302517</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-03-01</td>
      <td>94869.490789</td>
      <td>89121.298723</td>
      <td>99797.427717</td>
      <td>94869.490789</td>
      <td>94869.490789</td>
      <td>37.306077</td>
      <td>37.306077</td>
      <td>37.306077</td>
      <td>37.306077</td>
      <td>37.306077</td>
      <td>37.306077</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>94906.796867</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-04-01</td>
      <td>95178.251864</td>
      <td>89987.904019</td>
      <td>101154.016322</td>
      <td>95178.251864</td>
      <td>95178.251864</td>
      <td>166.278079</td>
      <td>166.278079</td>
      <td>166.278079</td>
      <td>166.278079</td>
      <td>166.278079</td>
      <td>166.278079</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>95344.529943</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1970-05-01</td>
      <td>95477.052904</td>
      <td>99601.487207</td>
      <td>110506.849617</td>
      <td>95477.052904</td>
      <td>95477.052904</td>
      <td>9672.619044</td>
      <td>9672.619044</td>
      <td>9672.619044</td>
      <td>9672.619044</td>
      <td>9672.619044</td>
      <td>9672.619044</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>105149.671948</td>
    </tr>
  </tbody>
</table>
</div>



There are many components in it but the main thing that you would care about is `yhat` which has the final predictions. `_lower` and `_upper` flags are for uncertainty intervals. 

- Final predictions: `yhat`, `yhat_lower`, and `yhat_upper`

Other columns are components that comprise the final prediction as we discussed in the introduction. Let's compare Prophet's components and what we see in our forecast DataFrame. 

$$y(t) = g(t) + s(t) + h(t) + e(t) $$

- Growth ($g(t)$): `trend`, `trend_lower`, and `trend_upper`
- Seasonality ($s(t)$): `additive_terms`, `additive_terms_lower`, and `additive_terms_upper`
    - Yearly seasonality: `yearly`, `yearly_lower`, and`yearly_upper`

The `additive_terms` represent the total seasonality effect, which is the same as yearly seasonality as we disabled weekly and daily seasonalities. All `multiplicative_terms` are zero because we used additive seasonality mode by default instead of multiplicative seasonality mode, which I will explain later.

Holiday effect ($h(t)$) is not present here as it's yearly data.

Prophet also has a nice built-in function for plotting each component. When we plot our forecast data, we see two 
components; general growth trend and yearly seasonality that appears throughout the years. If we had more components 
such as weekly or daily seasonality, they would have been presented here as well.


```python
# plot components
fig = m.plot_components(forecast)
```


<div style="text-align:center">
<img src="{{site.baseurl}}/images/2020-12-15-prophet-intro/output_25_0.png">
</div>
<br>


<a id="eval"></a>
# 4. Evaluate 

## 4.1. Evaluate the model on one test set

How good is our model? One way we can understand the model performance, in this case, is to simply calculate the 
root mean squared error (RMSE) between the actual and predicted values of the above test period.


```python
from statsmodels.tools.eval_measures import rmse

predictions = forecast.iloc[-len(test):]['yhat']
actuals = test['y']

print(f"RMSE: {round(rmse(predictions, actuals))}")
```

<pre class="output">
    RMSE: 32969.0
</pre>


However, this probably under-represents the general model performance because our data has a drastic change in the 
middle of the test period which is a pattern that has never been seen before. If our data was until 2019, the model 
performance score would have been much higher. 

    
<div style="text-align:center">
<img src="{{site.baseurl}}/images/2020-12-15-prophet-intro/output_31_0.png">
</div>
<br>


## 4.2. Cross validation

Alternatively, we can perform cross-validation. As previously discussed, time-series analysis strictly uses train 
data whose time range is earlier than that of test data. Below is an example where we use 5 years of train data to 
predict 1-year of test data. Each cut-off point is equally spaced with 1 year gap.


<div style="text-align:center">
<img src='{{site.baseurl}}/images/2020-12-15-prophet-intro/prophet_cv.png' alt="cv"/>
<figcaption>Time-series cross validation
</figcaption>
</div>
<br>

Prophet also provides built-in model diagnostics tools to make it easy to perform this cross-validation. You just 
need to define three parameters: horizon, initial, and period. The latter two are optional.

* horizon: test period of each fold  
* initial: minimum training period to start with
* period: time gap between cut-off dates

Make sure to define these parameters in string and in this format: 'X unit'. X is the number and unit is 'days' or 
'secs', etc. that is compatible with `pd.Timedelta`. For example, `10 days`.

You can also define `parallel` to make the cross validation faster.


```python
from fbprophet.diagnostics import cross_validation 

# test period
horizon = '365 days'

# itraining period (optional. default is 3x of horizon)
initial = str(365 * 5) + ' days'  

# spacing between cutoff dates (optional. default is 0.5x of horizon)
period = '365 days' 

df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel='processes')
```

<pre class="output">
    INFO:fbprophet:Making 43 forecasts with cutoffs between 1975-12-12 00:00:00 and 2017-12-01 00:00:00
    INFO:fbprophet:Applying in parallel with <concurrent.futures.process.ProcessPoolExecutor object at 0x12fb4d3c8>
</pre>


This is the predicted output using cross-validation. There can be many predictions for the same timestamp if `period`
 is smaller than `horizon`.


```python
# predicted output using cross validation
df_cv
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
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>y</th>
      <th>cutoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1976-01-01</td>
      <td>102282.737592</td>
      <td>100862.769604</td>
      <td>103589.684840</td>
      <td>102460.0</td>
      <td>1975-12-12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976-02-01</td>
      <td>96811.141761</td>
      <td>95360.095284</td>
      <td>98247.364027</td>
      <td>98528.0</td>
      <td>1975-12-12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1976-03-01</td>
      <td>112360.483572</td>
      <td>110908.136982</td>
      <td>113775.264669</td>
      <td>114284.0</td>
      <td>1975-12-12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976-04-01</td>
      <td>112029.016859</td>
      <td>110622.916037</td>
      <td>113458.999123</td>
      <td>117014.0</td>
      <td>1975-12-12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-05-01</td>
      <td>119161.998160</td>
      <td>117645.653475</td>
      <td>120579.267732</td>
      <td>123278.0</td>
      <td>1975-12-12</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>511</th>
      <td>2018-08-01</td>
      <td>279835.003826</td>
      <td>274439.830747</td>
      <td>285259.974314</td>
      <td>284989.0</td>
      <td>2017-12-01</td>
    </tr>
    <tr>
      <th>512</th>
      <td>2018-09-01</td>
      <td>261911.246557</td>
      <td>256328.677902</td>
      <td>267687.122886</td>
      <td>267434.0</td>
      <td>2017-12-01</td>
    </tr>
    <tr>
      <th>513</th>
      <td>2018-10-01</td>
      <td>268979.448383</td>
      <td>263001.411543</td>
      <td>274742.978202</td>
      <td>281382.0</td>
      <td>2017-12-01</td>
    </tr>
    <tr>
      <th>514</th>
      <td>2018-11-01</td>
      <td>255612.520483</td>
      <td>249813.339845</td>
      <td>261179.979649</td>
      <td>260473.0</td>
      <td>2017-12-01</td>
    </tr>
    <tr>
      <th>515</th>
      <td>2018-12-01</td>
      <td>257049.510224</td>
      <td>251164.508448</td>
      <td>263062.671327</td>
      <td>270370.0</td>
      <td>2017-12-01</td>
    </tr>
  </tbody>
</table>
<p>516 rows × 6 columns</p>
</div>



Below are different performance metrics for different rolling windows. As we did not define any rolling window, Prophet
 went ahead and calculated many different combinations and stacked them up in rows (e.g. 53 days, ..., 365 days). Each 
 metric is first calculated within each rolling window and then averaged across many available windows. 


```python
from fbprophet.diagnostics import cross_validation, performance_metrics 

# performance metrics  
df_metrics = performance_metrics(df_cv)  # can define window size, e.g. rolling_window=365
df_metrics
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
      <th>horizon</th>
      <th>mse</th>
      <th>rmse</th>
      <th>mae</th>
      <th>mape</th>
      <th>mdape</th>
      <th>coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>53 days</td>
      <td>3.886562e+07</td>
      <td>6234.229883</td>
      <td>5143.348348</td>
      <td>0.030813</td>
      <td>0.027799</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>1</th>
      <td>54 days</td>
      <td>3.983610e+07</td>
      <td>6311.584390</td>
      <td>5172.484468</td>
      <td>0.030702</td>
      <td>0.027799</td>
      <td>0.372549</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55 days</td>
      <td>4.272605e+07</td>
      <td>6536.516453</td>
      <td>5413.997433</td>
      <td>0.031607</td>
      <td>0.030305</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56 days</td>
      <td>4.459609e+07</td>
      <td>6678.030078</td>
      <td>5662.344846</td>
      <td>0.032630</td>
      <td>0.031911</td>
      <td>0.313725</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57 days</td>
      <td>4.341828e+07</td>
      <td>6589.254589</td>
      <td>5650.202377</td>
      <td>0.032133</td>
      <td>0.031481</td>
      <td>0.313725</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>361 days</td>
      <td>2.880647e+07</td>
      <td>5367.165528</td>
      <td>3960.025025</td>
      <td>0.020118</td>
      <td>0.015177</td>
      <td>0.607843</td>
    </tr>
    <tr>
      <th>116</th>
      <td>362 days</td>
      <td>3.158472e+07</td>
      <td>5620.028791</td>
      <td>4158.035261</td>
      <td>0.020836</td>
      <td>0.015177</td>
      <td>0.588235</td>
    </tr>
    <tr>
      <th>117</th>
      <td>363 days</td>
      <td>3.798731e+07</td>
      <td>6163.384773</td>
      <td>4603.360382</td>
      <td>0.022653</td>
      <td>0.017921</td>
      <td>0.549020</td>
    </tr>
    <tr>
      <th>118</th>
      <td>364 days</td>
      <td>4.615621e+07</td>
      <td>6793.836092</td>
      <td>4952.443173</td>
      <td>0.023973</td>
      <td>0.018660</td>
      <td>0.529412</td>
    </tr>
    <tr>
      <th>119</th>
      <td>365 days</td>
      <td>5.428934e+07</td>
      <td>7368.129817</td>
      <td>5262.131511</td>
      <td>0.024816</td>
      <td>0.018660</td>
      <td>0.529412</td>
    </tr>
  </tbody>
</table>
<p>120 rows × 7 columns</p>
</div>



<a id="trend"></a>
# 5. Trend Change Points

Another interesting functionality of `Prophet` is `add_changepoints_to_plot`. As we discussed in the earlier sections, there are a couple of points where the growth rate changes. Prophet can find those points automatically and plot them!


```python
from fbprophet.plot import add_changepoints_to_plot

# plot change points
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
```


    
<div style="text-align:center">
<img src="{{site.baseurl}}/images/2020-12-15-prophet-intro/output_42_0.png">
</div>
<br>
    



<a id=season></a>
# 6. Seasonality Mode

The growth in trend can be additive (rate of change is linear) or multiplicative (rate changes over time). When you 
see the original data, the amplitude of seasonality changes - smaller in the early years and 
bigger in the later years. So, this would be a `multiplicative` growth case rather than an `additive` growth case. We 
can adjust the `seasonality` parameter so we can take into account this effect. 


```python
# additive mode
m = Prophet(seasonality_mode='additive')
# multiplicative mode
m = Prophet(seasonality_mode='multiplicative')
```

You can see that the blue lines (predictions) are more in line with the black dots (actuals) when in multiplicative 
seasonality mode. 
    
<div style="text-align:center">
<img src="{{site.baseurl}}/images/2020-12-15-prophet-intro/output_44_1.png">
</div>
<br>
    


<div style="text-align:center">
<img src="{{site.baseurl}}/images/2020-12-15-prophet-intro/output_45_1.png">
</div>
<br>
    


<a id=save></a>
# 7. Saving Model

We can also easily export and load the trained model as json.


```python
import json
from fbprophet.serialize import model_to_json, model_from_json

# Save model
with open('serialized_model.json', 'w') as fout:
    json.dump(model_to_json(m), fout)

# Load model
with open('serialized_model.json', 'r') as fin:
    m = model_from_json(json.load(fin))  
```

<a id=ref></a>
# 8. References

- [Prophet documentation](https://facebook.github.io/prophet/docs/quick_start.html#python-api)
- [Prophet GitHub repository](https://github.com/facebook/prophet)
- [Prophet paper: Forecasting at scale](https://peerj.com/preprints/3190/)
- [Prophet in R](https://cran.r-project.org/web/packages/prophet/prophet.pdf)
- [U.S. traffic volume data](https://fred.stlouisfed.org/series/TRFVOLUSM227NFWA)
- [Python for Time Series Data Analysis](https://www.udemy.com/course/python-for-time-series-data-analysis/)
