import pandas as pd
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/10_1_1.csv")
df = pd.DataFrame(df)
print(df.head())

print((df.groupby('소주제')['정답여부'].sum()/df.groupby('소주제')['정답여부'].count()).sort_values(ascending = False))
print('경제')


df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/10_1_2.csv")
df = pd.DataFrame(df)
print(df.head())

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

print(df.groupby(['year', 'month'])['price'].sum().sort_values(ascending = False).index[1])
print('2023년 10월')

year, month = df.groupby(['year', 'month'])['price'].sum().sort_values(ascending = False).index[3]
print(df[(df['year'] == year) & (df['month'] == month)].groupby('category')['price'].sum().max())

df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/10_1_3.csv")
df = pd.DataFrame(df)
print(df.head())

df['word_count'] = df['message'].str.split(' ').str.len()
mean = df.groupby('label')['word_count'].mean()
print(mean)
print(f"{np.abs(mean['ham'] - mean['spam']):.3f}")

train = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/10_2_train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/10_2_test.csv")
print(train.head())

X = train.drop(columns = 'gas_totl')
y = train['gas_totl']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

num_columns = X.select_dtypes('number').columns.tolist()
cat_columns = X.select_dtypes('object').columns.tolist()

num_preprocess = make_pipeline(SimpleImputer(strategy = 'mean'))
cat_preprocess = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                               OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))

preprocess = ColumnTransformer([
    ('num', num_preprocess, num_columns),
    ('cat', cat_preprocess, cat_columns)
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('regressor', RandomForestRegressor(random_state = 42))
])


train_X , valid_X, train_y, valid_y = train_test_split(
    X, y, test_size = 0.3, random_state = 42
)

pipe.fit(train_X, train_y)
pred = pipe.predict(valid_X)
from sklearn.metrics import mean_squared_error
print("RMSE: ", np.sqrt(mean_squared_error(valid_y, pred)))

pipe.fit(X, y)
test_X = test.drop(columns = 'gas_totl', errors = 'ignore')
pred = pipe.predict(test_X)
pred = pd.DataFrame(pred, columns = ['pred'])
pred.to_csv('result.csv', index = False)

from scipy import stats
import statsmodels.api as sm
df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/10_3_1.csv")
df = pd.DataFrame(df)
print(df.head())

X = df.drop(columns = 'attrition')
y = df['attrition']
X = sm.add_constant(X)

model = sm.Logit(y, X).fit()
pvalues = model.pvalues.drop('const')
sig_pval_params = model.params.drop('const')[pvalues < 0.05]
for i in (sig_pval_params):
    print(f'{i:.3f}')

print('3-1-2 모름')
new_data = pd.DataFrame({'age': [20], 'income' : [3000], 'overtime' : [2]})
new_data = sm.add_constant(new_data, has_constant = 'add')
print(f"{model.predict(new_data).iloc[0]:.3f}")

df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/exam/10_3_2.csv")
df = pd.DataFrame(df)
print(df.head())

X = df.drop(columns = 'price')
y = df['price']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
pvalues = model.pvalues.drop('const')
sig_pval_params = model.params.drop('const')[pvalues < 0.05].sum()
print(f"{sig_pval_params:.3f}")
sig_vars = pvalues[pvalues < 0.05].index
X = X[list(sig_vars)]
y = df['price']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(f"{model.rsquared:.3f}")

new_data = pd.DataFrame({'area' : [100], 'height' : [10], 'wall': [1]})
new_data = new_data[list(sig_vars)]
new_data = sm.add_constant(new_data, has_constant = 'add')
print(f"{model.predict(new_data).iloc[0]:.3f}")