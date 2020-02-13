# Winning a Kaggle Competition in Python
[TOC]

[有目錄版HackMD筆記](https://hackmd.io/DnuI-2asQKmt-VRpsUBLew?both)

Resource: [Winning a Kaggle Competition in Python](https://www.datacamp.com/courses/winning-a-kaggle-competition-in-python)
## 我不知道的功能
- isin
`train=data[data.id.isin(train_id)]`

- pd.to_datetime()
![](https://i.imgur.com/vqhXXIU.png)

- itertool
![](https://i.imgur.com/6TZmeLj.png)



## intro
### submission
![](https://i.imgur.com/N7NpuY1.png)
![](https://i.imgur.com/4SRy30G.png)
![](https://i.imgur.com/4YeDweA.png)
![](https://i.imgur.com/zhE5Hwq.png)
![](https://i.imgur.com/hNKN3SI.png)
![](https://i.imgur.com/yYzHo7R.png)

- Public vs Private leaderboard
 ![](https://i.imgur.com/9tGPdec.png)
 ![](https://i.imgur.com/ZvXTIFk.png)
![](https://i.imgur.com/gGsM4jL.png)
![](https://i.imgur.com/5Gfok9K.png)
![](https://i.imgur.com/thl1TDC.png)
![](https://i.imgur.com/0dmUF1r.png)

### XGBoost
![](https://i.imgur.com/ilSO7sV.png)

- 比較複雜度跟overfitting
![](https://i.imgur.com/emYz1h6.png)
![](https://i.imgur.com/M9ZPj2w.png)

## Understand the problem
- EDA:exploratory data analysis
- local validation strategy:goal is to prevent overfitting
![](https://i.imgur.com/abMiDjw.png)
![](https://i.imgur.com/NODhbDn.png)
- custom error function
 ![](https://i.imgur.com/c4P36xL.png)
 
### Initial EDA
![](https://i.imgur.com/pMQRGom.png)

- 此course使用的competition
 ![](https://i.imgur.com/nWjGXYo.png)
 
- EDA
![](https://i.imgur.com/PmxbYhe.png)
![](https://i.imgur.com/aMJx5DP.png)
![](https://i.imgur.com/wOQhQ7T.png)
![](https://i.imgur.com/29RTfbT.png)
![](https://i.imgur.com/r2XwED2.png)
![](https://i.imgur.com/XvhfjAQ.png)
![](https://i.imgur.com/cfCoJzG.png)
![](https://i.imgur.com/SspSVt4.png)


### EDA常用code
```python
# Shapes of train and test data
print('Train shape:', train.shape)
print('Test shape:', test.shape)

# Train head()
print(train.head())

# Describe the target variable
print(train.fare_amount.describe())

# Train distribution of passengers within rides
print(train.passenger_count.value_counts())

```


### Local validation
![](https://i.imgur.com/ynjCHDE.png)
![](https://i.imgur.com/QgDwFDx.png)
![](https://i.imgur.com/24mv4su.png)
![](https://i.imgur.com/GWGGPTK.png)

- 每個fold的資料分布與原始資料相同，對於資料分布很不均或資料量小時有用 
![](https://i.imgur.com/wQ7JJCp.png)
![](https://i.imgur.com/fhkyFLg.png)
- code:K-fold
![](https://i.imgur.com/durpBrb.png)
![](https://i.imgur.com/Or6FqyR.png)
- code:strtified K-fold
![](https://i.imgur.com/wtPb09K.png)
![](https://i.imgur.com/yIXSjpP.png)
### validation usage
![](https://i.imgur.com/96P2P71.png)
![](https://i.imgur.com/h33dYTN.png)
![](https://i.imgur.com/1h2kc1P.png)
![](https://i.imgur.com/EmLUq7f.png)
![](https://i.imgur.com/xSRPkQK.png)
![](https://i.imgur.com/RVeQMcT.png)
![](https://i.imgur.com/37Udp85.png)
![](https://i.imgur.com/NeJNPiW.png)
![](https://i.imgur.com/xMRR2fH.png)

### code
```python
# Create TimeSeriesSplit object
time_kfold = TimeSeriesSplit(n_splits=3)

# Sort train data by date
train = train.sort_values('date')

# Iterate through each split
fold = 0
for train_index, test_index in time_kfold.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    
    print('Fold :', fold)
    print('Train date range: from {} to {}'.format(cv_train.date.min(), cv_train.date.max()))
    print('Test date range: from {} to {}\n'.format(cv_test.date.min(), cv_test.date.max()))
    fold += 1
```
![](https://i.imgur.com/TAydJiE.png)

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Sort train data by date
train = train.sort_values('date')

# Initialize 3-fold time cross-validation
kf = TimeSeriesSplit(n_splits=3)

# Get MSE scores for each cross-validation split
mse_scores = get_fold_mse(train, kf)

print('Mean validation MSE: {:.5f}'.format(np.mean(mse_scores)))
print('MSE by fold: {}'.format(mse_scores))
print('Overall validation MSE: {:.5f}'.format(np.mean(mse_scores) + np.std(mse_scores)))
```
![](https://i.imgur.com/skDQRr5.png)

## feature engineering
![](https://i.imgur.com/Vx8VlzP.png)
![](https://i.imgur.com/r7ss77T.png)
![](https://i.imgur.com/exBBBrO.png)
![](https://i.imgur.com/BoNVy9L.png)
![](https://i.imgur.com/3Rb2bqd.png)
![](https://i.imgur.com/CO8VbTl.png)
![](https://i.imgur.com/AuBdk1M.png)
![](https://i.imgur.com/3fRHyi4.png)

### feature engineering code
```python
# Look at the initial RMSE
print('RMSE before feature engineering:', get_kfold_rmse(train))

# Find the total area of the house
train['TotalArea'] = train['TotalBsmtSF'] + train['FirstFlrSF'] + train['SecondFlrSF']
print('RMSE with total area:', get_kfold_rmse(train))

# Find the area of the garden
train['GardenArea'] = train['LotArea'] - train['FirstFlrSF']
print('RMSE with garden area:', get_kfold_rmse(train))

# Find total number of bathrooms
train['TotalBath'] = train["FullBath"] + train["HalfBath"]
print('RMSE with number of bathrooms:', get_kfold_rmse(train))
```
![](https://i.imgur.com/YxRDbMn.png)

```python
# Concatenate train and test together
taxi = pd.concat([train, test])

# Convert pickup date to datetime object
taxi['pickup_datetime'] = pd.to_datetime(taxi['pickup_datetime'])

# Create a day of week feature
taxi['dayofweek'] = taxi['pickup_datetime'].dt.dayofweek

# Create an hour feature
taxi['hour'] = taxi['pickup_datetime'].dt.hour

# Split back into train and test
new_train = taxi[taxi['id'].isin(train['id'])]
new_test = taxi[taxi['id'].isin(test['id'])]
```

### categorical feature
![](https://i.imgur.com/5wNbUNq.png)
![](https://i.imgur.com/Lxj6qKW.png)
![](https://i.imgur.com/M106TNX.png)
![](https://i.imgur.com/rcS264U.png)
![](https://i.imgur.com/tlB1stV.png)
![](https://i.imgur.com/sEBa0Ku.png)

### categorical feature code
```python
# Concatenate train and test together
houses = pd.concat([train, test])

# Label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Create new features
houses['RoofStyle_enc'] = le.fit_transform(houses["RoofStyle"])
houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

# Look at new features
print(houses[['RoofStyle', 'RoofStyle_enc', 'CentralAir', 'CentralAir_enc']].head())
```
- one-hot encoding
```python
# Concatenate train and test together
houses = pd.concat([train, test])

# Label encode binary 'CentralAir' feature
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

# Create One-Hot encoded features
ohe = pd.get_dummies(houses['RoofStyle'], prefix='RoofStyle')

# Concatenate OHE features to houses
houses = pd.concat([houses, ohe], axis=1)

# Look at OHE features
print(houses[[col for col in houses.columns if 'RoofStyle' in col]].head(3))

```
![](https://i.imgur.com/Y3G05ny.png)

### target encoding
![](https://i.imgur.com/J64eRuO.png)
![](https://i.imgur.com/pKcct8r.png)
![](https://i.imgur.com/zuUfpoX.png)
![](https://i.imgur.com/Y9z4VXP.png)
- fold 1使用fold2的分布，fold 2使用fold 1的分布
![](https://i.imgur.com/9DHlfd6.png)
- 除非category number很大，不然一般會使用global mean來讓數值比較smooth (概念上類似於ML的smooth)
![](https://i.imgur.com/JrbH05a.png)
![](https://i.imgur.com/ZiXEJeZ.png)

### target encoding code
- You need to treat new categories in the test data. So, pass a global mean as an argument to the fillna() method.
```python
def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()
    
    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()
    
    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size +alpha)
    
    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values
```
```python
def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index)
    
    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
      
        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)
        
        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature       
    return train_feature.values
```
```python
def mean_target_encoding(train, test, target, categorical, alpha=5):
  
    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)
  
    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)
    
    # Return new features to add to the model
    return train_feature, test_feature
```
```python
# Create 5-fold cross-validation
kf = KFold(n_splits=5, random_state=123, shuffle=True)

# For each folds split
for train_index, test_index in kf.split(bryant_shots):
    cv_train, cv_test = bryant_shots.iloc[train_index], bryant_shots.iloc[test_index]

    # Create mean target encoded feature
    cv_train['game_id_enc'], cv_test['game_id_enc'] = mean_target_encoding(train=cv_train,
                                                                           test= cv_test,
                                                                           target='shot_made_flag',
                                                                           categorical='game_id',
                                                                           alpha=5)
    # Look at the encoding
    print(cv_train[['game_id', 'shot_made_flag', 'game_id_enc']].sample(n=1))
```
```python
# Create mean target encoded feature
train['RoofStyle_enc'], test['RoofStyle_enc'] = mean_target_encoding(train=train,
                                                                     test=test,
                                                                     target="SalePrice",
                                                                     categorical="RoofStyle",
                                                                     alpha=10)

# Look at the encoding
print(test[['RoofStyle', 'RoofStyle_enc']].drop_duplicates())
```
![](https://i.imgur.com/wHWQgJr.png)

### missing data
![](https://i.imgur.com/bbvfKjv.png)
![](https://i.imgur.com/3nK09NF.png)
![](https://i.imgur.com/65Je7hE.png)
![](https://i.imgur.com/XtYxMrD.png)
![](https://i.imgur.com/j7lNvZR.png)
![](https://i.imgur.com/suVJmWj.png)
![](https://i.imgur.com/bZtZFYs.png)
![](https://i.imgur.com/7PZaobp.png)
![](https://i.imgur.com/m8hcRuF.png)
![](https://i.imgur.com/6Ycukjv.png)

### missing data code
```python
# Read dataframe
twosigma = pd.read_csv( "twosigma_train.csv")

# Find the number of missing values in each column
print(twosigma.isnull().sum())
```
![](https://i.imgur.com/Ap2rqeU.png)

```python
# Import SimpleImputer
from sklearn.impute import SimpleImputer

# Create constant imputer
constant_imputer = SimpleImputer(strategy='constant', fill_value="MISSING" )

# building_id imputation
rental_listings[['building_id']] = constant_imputer.fit_transform(rental_listings[['building_id']])
```

## modeling
### baseline model
![](https://i.imgur.com/ajxjlhy.png)
![](https://i.imgur.com/9iKltdL.png)
![](https://i.imgur.com/GKE6zqH.png)
![](https://i.imgur.com/zoq2dyS.png)
![](https://i.imgur.com/olFH1kn.png)
![](https://i.imgur.com/KSJ8TfQ.png)
![](https://i.imgur.com/RsibetR.png)
![](https://i.imgur.com/7Nuw21Z.png)
### baseline model code
- mean baseline
```python
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate the mean fare_amount on the validation_train data
naive_prediction = np.mean(validation_train["fare_amount"])

# Assign naive prediction to all the holdout observations
validation_test['pred'] = naive_prediction

# Measure the local RMSE
rmse = sqrt(mean_squared_error(validation_test['fare_amount'], validation_test['pred']))
print('Validation RMSE for Baseline I model: {:.3f}'.format(rmse))
```
- groudby baseline
```python
# Get pickup hour from the pickup_datetime column
train['hour'] = train['pickup_datetime'].dt.hour
test['hour'] = test['pickup_datetime'].dt.hour

# Calculate average fare_amount grouped by pickup hour 
hour_groups = train.groupby('hour')['fare_amount'].mean()

# Make predictions on the test set
test['fare_amount'] = test.hour.map(hour_groups)

# Write predictions
test[['id','fare_amount']].to_csv('hour_mean_sub.csv', index=False)
```
- gradient boost baseline
```python
from sklearn.ensemble import RandomForestRegressor

# Select only numeric features
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
            'dropoff_latitude', 'passenger_count', 'hour']

# Train a Random Forest model
rf = RandomForestRegressor()
rf.fit(train[features], train.fare_amount)

# Make predictions on the test data
test['fare_amount'] = rf.predict(test[features])

# Write predictions
test[['id','fare_amount']].to_csv('rf_sub.csv', index=False)
```

### hyperparameter tuning
![](https://i.imgur.com/8zs8jQ4.png)
![](https://i.imgur.com/vxeAJRA.png)
![](https://i.imgur.com/jpqBuSb.png)
![](https://i.imgur.com/fc9JFME.png)
![](https://i.imgur.com/oGsCRzh.png)

### grid searching code
- You're given a function get_cv_score(), which takes the train dataset and dictionary of the model parameters as arguments and returns the overall validation RMSE score over 3-fold cross-validation.
```python
# Possible max depth values
max_depth_grid = [3,6,9,12,15]
results = {}

# For each value in the grid
for max_depth_candidate in max_depth_grid:
    # Specify parameters for the model
    params = {'max_depth': max_depth_candidate}

    # Calculate validation score for a particular hyperparameter
    validation_score = get_cv_score(train, params)

    # Save the results for each max depth value
    results[max_depth_candidate] = validation_score   
print(results)
```
![](https://i.imgur.com/qf5pSc9.png)

```python
import itertools

# Hyperparameter grids
max_depth_grid = [3,5,7]
subsample_grid = [0.8,0.9,1.0]
results = {}

# For each couple in the grid
for max_depth_candidate, subsample_candidate in itertools.product(max_depth_grid, subsample_grid):
    params = {'max_depth': max_depth_candidate,
              'subsample': subsample_candidate}
    validation_score = get_cv_score(train, params)
    # Save the results for each couple
    results[(max_depth_candidate, subsample_candidate)] = validation_score   
print(results)
```
![](https://i.imgur.com/LWaLqN0.png)

### model ensembling
![](https://i.imgur.com/oy6DFFB.png)
![](https://i.imgur.com/6Lj4vir.png)
![](https://i.imgur.com/uYDrwsh.png)
![](https://i.imgur.com/ml6yqvr.png)
![](https://i.imgur.com/jDhHF4C.png)
![](https://i.imgur.com/A7ZP2eA.png)
![](https://i.imgur.com/6QNOHIh.png)
![](https://i.imgur.com/aYiwvaF.png)

### model ensembling code
- blend
```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Train a Gradient Boosting model
gb = GradientBoostingRegressor().fit(train[features], train.fare_amount)

# Train a Random Forest model
rf = RandomForestRegressor().fit(train[features], train.fare_amount)

# Make predictions on the test data
test['gb_pred'] = gb.predict(test[features])
test['rf_pred'] = rf.predict(test[features])

# Find mean of model predictions
test['blend'] = (test['gb_pred'] + test['rf_pred']) / 2
print(test[['gb_pred', 'rf_pred', 'blend']].head(3))
```

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# Split train data into two parts
part_1, part_2 = train_test_split(train, test_size=0.5, random_state=123)

# Train a Gradient Boosting model on Part 1
gb = GradientBoostingRegressor().fit(part_1[features], part_1.fare_amount)

# Train a Random Forest model on Part 1
rf = RandomForestRegressor().fit(part_1[features], part_1.fare_amount)

# Make predictions on the Part 2 data
part_2['gb_pred'] = gb.predict(part_2[features])
part_2['rf_pred'] = rf.predict(part_2[features])

# Make predictions on the test data
test['gb_pred'] = gb.predict(test[features])
test['rf_pred'] = rf.predict(test[features])

from sklearn.linear_model import LinearRegression

# Create linear regression model without the intercept
lr = LinearRegression(fit_intercept=False)

# Train 2nd level model on the Part 2 data
lr.fit(part_2[['gb_pred', 'rf_pred']], part_2.fare_amount)

# Make stacking predictions on the test data
test['stacking'] = lr.predict(test[['gb_pred', 'rf_pred']])

# Look at the model coefficients
print(lr.coef_)
```
### Final tips and thoughts
![](https://i.imgur.com/aPlZGSy.png)
![](https://i.imgur.com/1wxa0M8.png)
![](https://i.imgur.com/5RpnaFR.png)
![](https://i.imgur.com/5Fd7vSe.png)
![](https://i.imgur.com/qsAm83b.png)
- final toughts
![](https://i.imgur.com/qJIuRYq.png)
![](https://i.imgur.com/WXl5qui.png)

```python

```

```python

```
