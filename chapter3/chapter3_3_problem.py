import pandas as pd
import numpy as np
train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/mroz_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/mroz_test.csv')
print(train.head())

train.info()
test.info()

X = train.drop(['lfp'], axis = 1)
y = train['lfp']

test_X = test.drop(['lfp'], axis = 1)

num_columns = X.select_dtypes('number').columns.tolist()
cat_columns = X.select_dtypes('object').columns.tolist()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

num_preprocess = make_pipeline(
    SimpleImputer(strategy='mean')
)

cat_preprocess = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore', sparse_output= False)
)

preprocess = ColumnTransformer([
    ('num', num_preprocess, num_columns),
    ('cat', cat_preprocess, cat_columns)
])

from sklearn.ensemble import RandomForestClassifier

full_pipe = Pipeline([
    ('preprocess', preprocess),
    ('classifier', RandomForestClassifier(random_state = 42))
])

from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42,
    stratift=y
)

full_pipe.fit(train_X, train_y)

from sklearn.metrics import f1_score

valid_pred = full_pipe.predict(valid_X)

print(f1_score(valid_y, valid_pred, average='macro'))

full_pipe.fit(X,y)
test_pred = full_pipe.predict(test_X)
test_pred = pd.DataFrame(test_pred, columns=['pred'])
test_pred.to_csv('result.csv', index = False)