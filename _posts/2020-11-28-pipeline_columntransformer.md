---
layout: post
title:  "Combining feature engineering and model fitting (Pipeline vs. ColumnTransformer)"
date:   2020-11-28 00:00:00 -0400
categories: python
---

<br>
<div style="text-align:center">
<img src="{{site.baseurl}}/images/pipeline_columntransformer/pipeline_unsplash.jpg" alt="drawing" />
<figcaption>Photo by <a href="https://unsplash.com/@spacexuan?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Crystal Kwok</a> on <a href="https://unsplash.com/s/photos/pipes?utm_source=unsplash&amp;utm_medium=referral&amp;utm_content=creditCopyText">Unsplash</a></figcaption>
</div>
<br>

In the [previous post]({% post_url 2020-11-21-Missing-data-imputation-using-sklearn %}), we learned about
 various missing data imputation strategies using 
scikit-learn. Before diving 
into finding the best imputation method for a given problem, I would like to first introduce two scikit-learn 
classes, `Pipeline` and `ColumnTransformer`. 

Both `Pipeline` amd `ColumnTransformer` are used to combine different transformers (i.e. feature engineering steps such
 as 
`SimpleImputer` and `OneHotEncoder`) to transform data. However, there are two major differences between them:


**1. `Pipeline` can be used for both/either of transformer and estimator (model) vs. `ColumnTransformer` is only for 
transformers**   
**2. `Pipeline` is sequential vs. `ColumnTransformer` is parallel/independent**

Don't worry if this sounds too complicated! I will walk you through what I mean by the above statements with code examples. I had a lot of fun while digging into these two classes, so I hope you enjoy and find it useful at the end as well! 

# Table of Conents
0. [Prepare Data](#prep)
1. [Put Transformers and an Estimator Together: Pipeline](#pipeline)
2. [Apply Transformers to Different Columns: ColumnTransformer](#columntransformer)
3. [Separate Feature Engineering Pipelines for Numerical and Categorical Variables](#separate)
4. [Final Pipeline](#final)
5. [Summary](#summary)
6. [References](#references)

<a id=prep></a>
# 0. Prepare Data

Let's first prepare the [house price data from Kaggle](https://www.kaggle
.com/c/house-prices-advanced-regression-techniques/data) we will be using in this post. The data is preprocessed by 
replacing `'?'` with `NaN`. Do not forget to split the data into train and test sets before performing any feature engineering steps to avoid data leakage!


```python
import pandas as pd 

# preparing data 
from sklearn.model_selection import train_test_split

# feature engineering: imputation, scaling, encoding
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# putting together in pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# model to use
from sklearn.linear_model import Lasso
```


```python
# import house price data 
df = pd.read_csv('../data/house_price/train.csv', index_col='Id')

# numerical columns vs. categorical columns 
num_cols = df.drop('SalePrice', axis=1).select_dtypes('number').columns
cat_cols = df.drop('SalePrice', axis=1).select_dtypes('object').columns

# split train and test dataset 
X_train, X_test, y_train, y_test = train_test_split(df.drop('SalePrice', axis=1), 
                                                    df['SalePrice'], 
                                                    test_size=0.3, 
                                                    random_state=0)

# check the size of train and test data
X_train.shape, X_test.shape
```




    ((1022, 79), (438, 79))




```python
X_train.head()
```



<div class="table-wrapper" markdown="block">
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
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
    <tr>
      <th>Id</th>
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
      <th>65</th>
      <td>60</td>
      <td>RL</td>
      <td>NaN</td>
      <td>9375</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>683</th>
      <td>120</td>
      <td>RL</td>
      <td>NaN</td>
      <td>2887</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>961</th>
      <td>20</td>
      <td>RL</td>
      <td>50.0</td>
      <td>7207</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1385</th>
      <td>50</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9060</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>10</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>30</td>
      <td>RL</td>
      <td>60.0</td>
      <td>8400</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Bnk</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 79 columns</p>
</div>

</div>



<a id=pipeline></a>
# 1. Put Transformers and an Estimator Together: Pipeline

Let's say we want to train a Lasso regression model that predicts `SalePrice`. Instead of using all of the 79 variables we have, let's use only numerical variables this time. 

I already know there is plenty of missing data in some columns (e.g. `LotFrontage`, `MasVnrArea`, and `GarageYrBlt` among numerical columns), so we want to perform missing data imputation before fitting a model. Also, let's say we also want to scale the data using `StandardScaler` because the scale of variables is all different.

This is what we would do normally to fit a model:


```python
# take only numerical data
X_temp = X_train[num_cols].copy()

# missing data imputation
imputer = SimpleImputer(strategy='mean')
X_impute = imputer.fit_transform(X_temp)  # np.ndarray
X_impute = pd.DataFrame(X_impute, columns=X_temp.columns)  # pd.DataFrame

# scale data 
scaler = StandardScaler()
X_scale = scaler.fit_transform(X_impute)  # np.ndarray
X_scale = pd.DataFrame(X_scale, columns=X_temp.columns)  # pd.DataFrame

# fit model
lasso = Lasso()
lasso.fit(X_scale, y_train)
lasso.score(X_scale, y_train)
```




    0.8419801151434141



This is great but we have to manually move data from one step to another: we pass the output of the first step (`SimpleImputer`) to the second step (`StandardScaler`) as an input (`X_impute`). And then, the output of the second step (`StandardScaler`) is passed to the third step (`Lasso`) as an input (`X_scale`). If we have more feature engineering steps, it will be more complex to handle different inputs and outputs. So, here `Pipeline` comes to the rescue!

**With `Pipeline`, you can combine transformers and an estimator (model) together**. You can transform your data and then fit a model with the transformed data. You just need to pass a list of tuples defining the steps in order: (step_name, transformer or estimator object). Let's rewrite the same logic using `Pipeline`.


```python
# define feature engineering and model together
pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                 ('scaler', StandardScaler()),
                 ('lasso', Lasso())])

# fit model
pipe.fit(X_temp, y_train)
pipe.score(X_temp, y_train)
```




    0.8419801151434141



Awesome! We saved a lot of lines and it looks much cleaner and more understandable! As you can see, **Pipeline passes 
the 
first step's output to the next step as its input, meaning Pipeline is sequential**.

<a id=columntransformer></a>
# 2. Apply Transformers to Different Columns: ColumnTransformer

Let's go back to our original dataset where we had both numerical and categorical variables. Because we cannot apply 
mean imputation to categorical variables (there is no 'mean' in categories!), we would want to use something different. One of the commonly used techniques is mode imputation (filling with the most frequent category), so let's use that. 

Mean imputation for numerical variables and mode imputation for categorical variables - can we do this in Pipeline as below?


```python
# Can we do this? 
pipe = Pipeline([('num_imputer', SimpleImputer(strategy='mean')),
                 ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                 ('lasso', Lasso())])

pipe.fit(X_train, y_train)
```

Unfortunately, no! If you run the above code, it will throw an error like this: 

```
ValueError: Cannot use mean strategy with non-numeric data:
could not convert string to float: 'RL'
```

The error happens when `Pipeline` attempts to apply mean imputation to all of the columns including a categorical 
variable that contains a string category called `'RL'`. Remember mean imputation can only be applied to numerical variables so our `SimpleImputer(strategy='mean')` freaked out! 

**We need to let our `Pipeline` know which columns to apply which transformer. How do we do that? We do it with `ColumnTransformer`!**

`ColumnTransformer` is similar to `Pipeline` in the sense that you put transformers together as a list of tuples, but 
in this time, you pass one more argument: a list of the column names you want to apply a transformer. 


```python
# applying different transformers to different columns 
transformer = ColumnTransformer(
    [('numerical', SimpleImputer(strategy='mean'), num_cols), 
     ('categorical', SimpleImputer(strategy='most_frequent'), cat_cols)])

# fit transformer with out train data
transformer.fit(X_train)

# transform the train data and create a DataFrame with the transformed data
X_train_transformed = transformer.transform(X_train)
X_train_transformed = pd.DataFrame(X_train_transformed, 
                                   columns=list(num_cols) + list(cat_cols))
```

You may have noticed we defined the output columns to be `list(num_cols) + list(cat_cols)`, not `X_train.columns`. This is because **`ColumnTransformer` fits each transformer independently in parallel and concatenates all of the outputs at the end**. 

That is, `ColumnTransformer` takes **only** numerical columns (`num_cols`), fits and transforms them using 
`SimpleImputer(strategy='mean')`, sets the output aside. At the same time, it does the same thing for categorical 
columns (`cat_cols`) with `SimpleImputer(strategy='most_frequent')`. When it is done with each and every step, it 
combines all of the two outputs in the order that the transformers are performed. Therefore, **be aware of the column orders because the final output may be different from your original DataFrame!**

Note that `ColumnTransformer` can only be used for transformers, not estimators. We cannot include `Lasso()` and fit the model as we did with `Pipeline`. **`ColumnTransformer` is only used for data pre-processing, so there is no `predict` or `score` as in `Pipeline`**. To train a model and calculate a performance score, we will need `Pipeline` again.

<a id=separate></a>
# 3. Separate Feature Engineering Pipelines for Numerical and Categorical Variables

Let's go one step further and include more feature engineering steps. In addition to the missing data imputation, we 
also want to scale our numerical variables using `StandardScaler` and encode the categorical variables using 
`OneHotEncoder`. Can we do something like this then?


```python
# Can we do this? 
transformer = ColumnTransformer(
    [('numerical_imputer', SimpleImputer(strategy='mean'), num_cols), 
     ('numerical_scaler', StandardScaler(), num_cols), 
     ('categorical_imputer', SimpleImputer(strategy='most_frequent'), cat_cols),
     ('categorical_encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)])

transformer.fit(X_train)
```

No! 

As we saw in the previous section, each step in `ColumnTransformer` is independent. Therefore, the input for the `OneHotEncoder()` is not the output of the `SimpleImputer(strategy='most_frequent')` but just a subset of the original DataFrame (`cat_cols`) which is not imputed. You cannot one-hot-encode a categorical variable that has missing data. 

We need something that can sequentially pass data throughout multiple feature engineering steps. Sequentially moving data... sounds familiar, right? Yes, you can do this with `Pipeline`! 

However, we need to create a feature engineering pipeline for numerical variables and categorical variables separately. So, we can come up with something like this:


```python
# feature engineering pipeline for numerical variables 
num_pipeline= Pipeline([('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())])

# feature engineering pipeline for categorical variables 
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
```

You can think it as creating a 'new transformer' that combines multiple transformers for each type of variable. Doesn't it sounds cool? 

<a id=final></a>
# 4. Final Pipeline

Okay. Now that we have feature engineering pipelines defined for both numerical variables and categorical variables, we can put things together to train a Lasso model using `ColumnTransformer` and `Pipeline`. 


```python
# put numerical and categorical feature engineering pipelines together
preprocessor = ColumnTransformer([("num_pipeline", num_pipeline, num_cols),
                                  ("cat_pipeline", cat_pipeline, cat_cols)])

# put transformers and an estimator together
pipe = Pipeline([('preprocessing', preprocessor),
                 ('lasso', Lasso(max_iter=10000))])  # increased max_iter to converge

# fit model 
pipe.fit(X_train, y_train)
pipe.score(X_train, y_train)
```




    0.9483539967729575



This is very neat! We applied different sets of feature engineering steps to numercial and categorical variables and then trained a model in only a few lines of code. 

Thinking of how long and complex the code would be without `ColumnTransformer` and `Pipeline`, aren't you tempted to try this out right now? 

<a id=summary></a>
# Summary 

In this post, we looked at how to combine feature engineering steps and a model fitting step together using `Pipeline` and `ColumnTransformer`. Especially we learned that we can use 
- `Pipeline` for combining transformers and an estimator 
- `ColumnTransformer` for applying different transformers to different columns 
- `Pipeline` for creating different feature engineering pipelines for numerical and categorical variables that sequentially apply a different set of transformers


Also, check out the table below to recap the differences between `Pipeline` vs. `ColumnTransformer`:

|   |    Pipeline | ColumnTransformer|
|---|-------------|------------------|
|Used for                   | Both/either of transformers and estimator | Transformers only|
|Main methods               | fit, transform, predict, and score| fit, and transform (no predict or score)|
|Can pick columns to apply  | No | Yes|
|Each step is performed     | Sequentially | Independently|
|Transformed output columns | Same as input | May differ depending on the defined steps|



<a id=References></a>
# References 
- [Pipeline, ColumnTransformer and FeatureUnion explained](https://towardsdatascience.com/pipeline-columntransformer-and-featureunion-explained-f5491f815f)
- [Feature Engineering for Machine Learning](https://www.udemy.com/course/feature-engineering-for-machine-learning/)
- [sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [sklearn ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)


