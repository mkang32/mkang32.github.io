---
layout: post
title: Finding The Best Feature Engineering Strategy Using sklearn GridSearchCV
date: 2020-12-05 22:49 -0500
---


<br>
<div style="text-align:center">
<img src="{{site.baseurl}}/images/search.jpg" alt="search"/>
<figcaption>Photo by <a href="https://unsplash.com/@laughayette?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Marten Newhall</a> on <a href="https://unsplash.com/s/photos/search?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></figcaption>
</div>
<br>




   
We previously reviewed a few missing data imputation strategies using sklearn in [this post]({% post_url 2020-11-21-Missing-data-imputation-using-sklearn %}), but which one should we use? How do we know which one works best for our data? Should we manually write a script to fit a model for each strategy and track the model performance? We could, but it would be a headache to track many different models, especially if we use cross validation to get more reliable experiment results. 
    
Fortunately, sklearn offers great tools to streamline and optimize the process, which are `GridSearchCV` and `Pipeline`! You might be already familiar with using `GridSearchCV` for finding optimal hyperparameters of a model, but you might not be familiar with using it for finding optimal feature engineering strategies. 
    
In this post, I would like to walk you through how `GridSearchCV` and `Pipeline` can be used to find the best feature engineering strategies for the given data. We will focus on missing data imputation strategies here but it can be used for any other feature engineering steps or combinations. 

# Table of Conents
1. [Prepare Data](#prep)
2. [Setup a Base Pipeline](#base)  
3. [Finding The Best Imputation Technique Using GridSearchCV](#find_best)
4. [References](#ref)


<a id='prep'></a>
# 1. Prepare Data

First, import necessary libraries and prepare data. We will use the [house price data from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) in this post.


```python
import pandas as pd 

# preparing data 
from sklearn.model_selection import train_test_split

# feature scaling, encoding
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# putting together in pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# model selection
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
```


```python
# import house price data 
df = pd.read_csv('../data/house_price/train.csv', index_col='Id')

# find numerical columns vs. categorical columns, except for the target ('SalePrice')
num_cols = df.drop('SalePrice', axis=1).select_dtypes('number').columns
cat_cols = df.drop('SalePrice', axis=1).select_dtypes('object').columns

# define X and y for GridSearchCV
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# split train and test dataset 
X_train, X_test, y_train, y_test = train_test_split(df.drop('SalePrice', axis=1), 
                                                    df['SalePrice'], 
                                                    test_size=0.3, 
                                                    random_state=0)
```

<a id='base'></a>
# 2. Setup a Base Pipeline


## 2.1. Define Pipelines


The next step is defining a base `Pipeline` for our model as below. 

1. Define two feature preprocessing pipelines; one for numerical variables (`num_pipe`) and the other for categorical variables (`cat_pipe`). `num_pipe` has `SimpleImputer` for missing data imputation and `StandardScaler` for scaling data. `cat_pipe` has `SimpleImputer` for missing data imputation and `OneHotEncoder` for encoding categorical data as numerical data. 

2. Combine those two pipelines together using `ColumnTransformer` to apply them to a different set of columns. 

3. Define the final pipeline called `pipe` by putting the `preprocess` pipeline together with an estimator, which is `Lasso` regression in this example. 


For details of this pipeline, please check out the previous post [Combining Feature Engineering and Model Fitting 
(Pipeline vs. ColumnTransformer)]({% post_url 
2020-11-28-pipeline_columntransformer %}). 



```python
# feature engineering pipeline for numerical variables 
num_pipe= Pipeline([('imputer', SimpleImputer(strategy='mean', add_indicator=False)),
                    ('scaler', StandardScaler())])

# feature engineering pipeline for categorical variables 
# Note: fill_value='Missing' is not used for strategy='most_frequent' but defined here for later use
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', fill_value='Missing')),
                     ('encoder', OneHotEncoder(handle_unknown='ignore'))])

# put numerical and categorical feature engineering pipelines together
preprocess = ColumnTransformer([("num_pipe", num_pipe, num_cols),
                                ("cat_pipe", cat_pipe, cat_cols)])


# put transformers and an estimator together
pipe = Pipeline([('preprocess', preprocess),
                 ('lasso', Lasso(max_iter=10000))]) 

```

## 2.2. Fit Pipeline

Okay, so let's fit the model with our train data and test with the test data. Here, we get 0.63 for the score, which is $R^2$ of the prediction in this case ([sklearn Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso.score)).


```python
# fit model 
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```



<pre class="output">
0.6308258188969262
</pre>


We could also cross validate the model using `cross_val_score`. It splits the whole data into 5 sets and calculates the score 5 times by fitting and testing with different sets each time.


```python
# cross validate
cross_val_score(pipe, X, y, cv=5)
```



<pre class="output">
array([0.85570392, 0.8228412 , 0.80381056, 0.88846653, 0.63236809])
</pre>

## 2.3. Diffferent Parameters To Test

Let's say we want to try different combinations of missing data imputation strategies for `SimpleImputer`, such as both `'median'` and `'mean'` for `strategy` and both `True` and `False` for `add_indicator`. To compare all of the cases, we need to test 4 different models with the following numerical variable imputation methods: 

    SimpleImputer(strategy='mean', add_indicator=False)
    SimpleImputer(strategy='median', add_indicator=False)
    SimpleImputer(strategy='mean', add_indicator=True)
    SimpleImputer(strategy='median', add_indicator=True)

We could copy and paste the script we wrote above, replace the corresponding step, and compare the performance of each case. It would not be too bad for the 4 combinations. But what if we want to test more combinations such as `strategy='constant'` and `strategy='most_frequent'` for categorical variables? Now it becomes 8 combinations ($2 \times 2 \times 2 = 8$). 

The more parameters we add, the more cases we have to test ad track (exponentially growing cases!). But don't worry! We have `GridSearchCV`. 


<a id='find_best'></a>
# 3. Finding The Best Imputation Technique Using GridSearchCV


## 3.1. What Is GridSearchCV? 
`GridSearchCV` is a sklearn class that is used to find parameters with the best cross validation given the search space (parameter combinations). This can be used not only for hyperparameter tuning for estimators (e.g. `alpha` for Lasso), but also for parameters in any preprocessing step. We just need to define parameters that we want to optimize and pass them to `GridSearchCV()` as a dictionary. 

The rule for defining the grid search parameter key-value pair is the following:
1. Key: a string that combines the name of the step with the name of the parameter with two understcores  
2. Value: a list of parameter values to test 

In short, it's `{'step_name__parameter_name': a list of values}`. For example, if the step name is `lasso` and the parameter name is `alpha`, your grid search param becomes: 
`{'lasso__alph': [1, 5, 10]}` 


## 3.2. Defining Nested Parameters
What about nested parameters that we have in our case? For example, our missing data imputation strategy for numerical variables is a few steps away from the final pipeline such as `preprocess` --> `num_pipe` --> `imputer`. 

Even for those cases, we can simply expand the key by keeping combining them with two unerstcore:

`{'preprocess__num_pipe__imputer__strategy': ['mean', 'median', 'most_frequent']}`

## 3.3. Defining and Fitting GridSearchCV

With the basics of `GridSearchCV`, let's define `GridSearchCV` and its parameters for our problem.


```python
# define the GridSearchCV parameters 
param = dict(preprocess__num_pipe__imputer__strategy=['mean', 'median', 'most_frequent'],
             preprocess__num_pipe__imputer__add_indicator=[True, False],
             preprocess__cat_pipe__imputer__strategy=['most_frequent', 'constant']) 

# define GridSearchCV 
grid_search = GridSearchCV(pipe, param)
```

Now it's time to find the best parameters by simply running `fit`!


```python
# search the best parameters by fitting the GridSearchCV 
grid_search.fit(X, y)
```




## 3.4. Checking the results

To check the combinations of parameters we tested and their performances in each cross validation set in terms of score and time, we can use the attribute `.cv_results`. 


```python
# check out the results
grid_search.cv_results_
```



So, which model did the `GridSearchCV` find to be most effective and what's its score? Let's check out `.bast_params_` and `.best_score_` attributes for that. 


```python
# check out the best parameter combination found
grid_search.best_params_
```



<pre class="output">
{'preprocess__cat_pipe__imputer__strategy': 'constant',
 'preprocess__num_pipe__imputer__add_indicator': False,
 'preprocess__num_pipe__imputer__strategy': 'most_frequent'}
</pre>


```python
# score 
grid_search.best_score_
```



<pre class="output">
0.8058139542143075
</pre>




Awesome! It seems like using a `constant` value for categorical variables and `most_frequent` values for numerical 
variables without missing indicator was found to be most effective in this case. Again, the best missing data imputation strategy depends on the data and the model. Try out with your data and see what works best for yours! 

<a id='ref'></a>
# 4. References
- [sklearn GridSearchCV]('https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html')
- [sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [sklearn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Feature Engineering for Machine Learning](https://www.udemy.com/course/feature-engineering-for-machine-learning/)



