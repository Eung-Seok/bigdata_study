import pandas as pd
import numpy as np

train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/prestige_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/prestige_test.csv')
print(train.info())
print(test.info())

from sklearn.model_selection import train_test_split
train_X = train.drop(['prestige','census'], axis = 1)
train_y = train['prestige']

test_X = test.drop(['prestige','census'], axis=1)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

num_preprocess = make_pipeline(SimpleImputer(strategy = 'mean'),
                               StandardScaler())

cat_preprocess = make_pipeline(OneHotEncoder(handle_unknown='ignore', sparse_output= False))

num_columns = train_X.select_dtypes('number').columns.tolist()
cat_columns = train_X.select_dtypes('object').columns.tolist()

preprocess = ColumnTransformer(
    [
        ('num', num_preprocess, num_columns),
        ('cat', cat_preprocess, cat_columns)
    ]
)

train_X, valid_X, train_y, valid_y = train_test_split(
    train_X,
    train_y,
    test_size = 0.3,
    random_state= 42
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = Pipeline(
    [
        ('preprocess', preprocess),
        ('regressor', RandomForestRegressor(random_state=42))
    ]
)

model.fit(train_X,train_y)
valid_pred = model.predict(valid_X)

print('교차검증 MSE: ', mean_squared_error(valid_y, valid_pred, squared = False))

test_pred = model.predict(test_X)
test_pred = pd.DataFrame(test_pred, columns = ['pred'])

test_pred.to_csv('result.csv', index = False)