import pandas as pd
import numpy as np

np.random.seed(111)

df = pd.DataFrame({
    'id': range(1, 121),
    'dept': np.random.choice(['HR', 'Sales', 'IT', 'Marketing'], 120),
    'grade': np.random.choice(['A', 'B', 'C'], 120),
    'join_date': pd.date_range('2018-01-01', periods=120, freq='20D'),
    'satisfaction': np.random.normal(72, 12, 120).round(1),
    'tenure': np.random.normal(5, 2, 120).round(1),
    'salary': np.random.normal(5200, 900, 120).round(0),
    'training_count': np.random.randint(0, 12, 120),
    'review': np.random.choice([
        'good fast kind',
        'late slow issue',
        'excellent support',
        'normal response',
        'bad delay complaint'
    ], 120)
})

df.loc[[5, 17, 44, 88], 'satisfaction'] = np.nan
df.loc[[9, 33, 71], 'tenure'] = np.nan
df.loc[[12, 52], 'salary'] = np.nan

print(df.head())
print(df.info())
# 1-1 satisfaction 결측치는 전체 평균으로 채우고, tenure 결측치가 있는 행은 삭제하시오.
# 이후 salary의 1사분위수를 구하시오.
# 소수 첫째 자리에서 반올림하여 정수로 출력하시오.

df['satisfaction'] = df['satisfaction'].fillna(df['satisfaction'].mean())
df = df.dropna(subset = 'tenure')
print(f"{df['salary'].quantile(0.25):.2f}")
# 1-2 join_date에서 연도와 월을 각각 추출하시오.
# 2020년 이후 입사자 중 training_count가 가장 큰 상위 10명의 평균 satisfaction을 구하시오.
# 소수 둘째 자리까지 출력하시오.
df['join_date'] = pd.to_datetime(df['join_date'])
df['month'] = df['join_date'].dt.month
df['year'] = df['join_date'].dt.year

print(df[df['year'] >= 2020]['training_count'].sort_values(ascending = True).head(10).mean())
# 1-3 review 컬럼에서 공백 기준 단어 수를 계산한 word_count 컬럼을 만드시오.
# dept별 평균 word_count를 구하고, 평균이 두 번째로 높은 부서명을 출력하시오.
df['word_count'] = df['review'].str.split(' ').str.len()
print(df.groupby('dept')['word_count'].mean().sort_values(ascending = False))
print('IT')
import pandas as pd
import numpy as np

np.random.seed(222)

n_train = 700
n_test = 300

train = pd.DataFrame({
    'ID': range(10000, 10000 + n_train),
    'biz_type': np.random.choice(['food', 'retail', 'office', 'medical'], n_train),
    'area': np.random.normal(120, 35, n_train).round(1),
    'age': np.random.randint(1, 40, n_train),
    'num_households': np.random.randint(5, 80, n_train),
    'floor': np.random.randint(1, 20, n_train),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu'], n_train)
})

test = pd.DataFrame({
    'ID': range(20000, 20000 + n_test),
    'biz_type': np.random.choice(['food', 'retail', 'office', 'medical', 'education'], n_test),
    'area': np.random.normal(120, 35, n_test).round(1),
    'age': np.random.randint(1, 40, n_test),
    'num_households': np.random.randint(5, 80, n_test),
    'floor': np.random.randint(1, 20, n_test),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Jeju'], n_test)
})

effect_biz = {'food': 180, 'retail': 120, 'office': 90, 'medical': 220}
effect_region = {'Seoul': 150, 'Busan': 80, 'Daegu': 40}

train['gas_tot'] = (
    500
    + train['area'] * 7.5
    - train['age'] * 8
    + train['num_households'] * 12
    + train['floor'] * 3
    + train['biz_type'].map(effect_biz)
    + train['region'].map(effect_region)
    + np.random.normal(0, 180, n_train)
).round(2)

train.loc[[10, 44, 120, 301], 'area'] = np.nan
train.loc[[7, 89], 'biz_type'] = np.nan
test.loc[[5, 66], 'area'] = np.nan
test.loc[[9], 'biz_type'] = np.nan

print(train.head())
print(test.head())
print(train.info())

# 학습용 데이터를 이용하여 gas_tot을 예측하는 모델을 만들고, 평가용 데이터에 적용하여 예측 결과를 제출하시오.

# 평가 지표: MAE
# 제출 파일명: result.csv
# 제출 컬럼: ID, pred
# pred: 예측된 gas_tot
# 제출 행 수는 평가용 데이터 행 수와 같아야 함

print(train.shape, train['ID'].nunique())
X1 = train.drop(['ID', 'gas_tot'], axis = 1)
y1 = train['gas_tot']

X2 = train.drop(columns = 'gas_tot')
y2 = train['gas_tot']
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

num_columns1 = X1.select_dtypes('number').columns.tolist()
cat_columns1 = X1.select_dtypes('object').columns.tolist()

num_columns2 = X2.select_dtypes('number').columns.tolist()
cat_columns2 = X2.select_dtypes('object').columns.tolist()

num_preprocess = make_pipeline(SimpleImputer(strategy = 'mean'))
cat_preprocess = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                               OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))

preprocess1 = ColumnTransformer([
    ('num', num_preprocess, num_columns1),
    ('cat', cat_preprocess, cat_columns1)
])

preprocess2 = ColumnTransformer([
    ('num', num_preprocess, num_columns2),
    ('cat', cat_preprocess, cat_columns2)
])

pipe1 = Pipeline([
    ('preprocess', preprocess1),
    ('regressor', RandomForestRegressor(random_state = 42))
])

pipe2 = Pipeline([
    ('preprocess', preprocess2),
    ('regressor', RandomForestRegressor(random_state = 42))
])

train_X1, valid_X1, train_y1, valid_y1 = train_test_split(
    X1,
    y1,
    test_size = 0.3,
    random_state = 42
)

train_X2, valid_X2, train_y2, valid_y2 = train_test_split(
    X2,
    y2,
    test_size = 0.3,
    random_state = 42
)

pipe1.fit(train_X1, train_y1)
pred1 = pipe1.predict(valid_X1)
pipe2.fit(train_X2, train_y2)
pred2 = pipe2.predict(valid_X2)
print('X에 ID제외한 MAE: ', mean_absolute_error(valid_y1, pred1))
print('X에 ID포함한 MAE: ', mean_absolute_error(valid_y2, pred2))

print('X에 ID를 제거한 MAE가 더 작게 나왔기 때문에, X에 ID를 제거한 모델을 채택한다.')
pipe1.fit(X1, y1)
test_X = test.drop(['ID', 'gas_tot'], axis = 1, errors = 'ignore')
pred = pipe1.predict(test_X)
ID = test['ID']
pred = pd.DataFrame({
    'ID': ID,
    'pred' : pred
})

pred.to_csv('result.csv', index = False)



import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(333)

df = pd.DataFrame({
    'Classification': np.repeat([1, 2], [55, 65]),
    'Resistin': np.concatenate([
        np.random.lognormal(mean=2.1, sigma=0.35, size=55),
        np.random.lognormal(mean=2.35, sigma=0.48, size=65)
    ])
})

print(df.head())
print(df['Classification'].value_counts())

# Resistin을 로그 변환한 값을 기준으로 Classification 값에 따라 두 집단으로 나누어라.
# 두 집단의 분산비 통계량을 구하시오.
# 단, 큰 분산을 작은 분산으로 나누시오.
# 소수 셋째 자리까지 출력하시오.
# 두 집단의 합동분산 추정량을 구하시오.
# 소수 셋째 자리까지 출력하시오.
# 두 집단의 로그 변환값 평균 차이에 대한 p-value를 구하시오.
# 2번에서 구한 합동분산을 사용하는 조건으로 계산하시오.
# 소수 셋째 자리까지 출력하시오.
df['Resistin'] = np.log(df['Resistin'])
first = df[df['Classification'] == 1]
second = df[df['Classification'] == 2]

first_var = first['Resistin'].var()
second_var = second['Resistin'].var()
stat = max(first_var, second_var) / min(first_var, second_var)
print(f"{stat:.3f}")
print('합동분산 공식 모름')
stat, pvalue = stats.ttest_ind(first['Resistin'], second['Resistin'], equal_var = True, alternative = 'two-sided')
print(f"{pvalue:.3f}")



import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(444)

df = pd.DataFrame({
    'study_time': np.random.normal(5, 1.5, 80),
    'score': np.random.normal(75, 10, 80)
})

df['score'] = (
    50
    + df['study_time'] * 5.2
    + np.random.normal(0, 6, 80)
).round(1)

print(df.head())
# study_time과 score의 상관계수를 구하시오.
# 소수 셋째 자리까지 출력하시오.
# 위 상관관계에 대한 p-value를 구하시오.
# 소수 셋째 자리까지 출력하시오.

stat, pvalue = stats.pearsonr(df['study_time'],df['score'])
print(f"{stat:.3f}")
print(f"{pvalue:.3f}")


import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(555)

n = 180

df = pd.DataFrame({
    'price': np.random.normal(30000, 5000, n),
    'area': np.random.normal(85, 20, n),
    'height': np.random.normal(12, 3, n),
    'wall': np.random.choice([0, 1], n)
})

df['price'] = (
    8000
    + df['area'] * 210
    + df['height'] * 350
    + df['wall'] * 1800
    + np.random.normal(0, 2500, n)
).round(1)

print(df.head())
# price를 종속변수로 하고 나머지 변수를 독립변수로 하는 모형을 적합하시오.
# 유의확률이 0.05 미만인 변수들의 회귀계수 합을 구하시오.
# 단, 절편은 제외한다.
# 소수 셋째 자리까지 출력하시오.
# 유의확률이 0.05 미만인 변수만 사용하여 다시 모형을 적합하시오.
# 결정계수를 구하시오.
# 소수 셋째 자리까지 출력하시오.
# area=100, height=10, wall=1인 데이터의 예측값을 구하시오.
# 단, 2번에서 선택된 변수만 사용한다.
# 소수 셋째 자리까지 출력하시오.

X = df.drop(columns = 'price')
y = df['price']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
pvalues = model.pvalues.drop('const')
print(f"{model.params.drop('const')[pvalues < 0.05].sum():.3f}")

sig_vars = pvalues[pvalues < 0.05].index
X = X[list(sig_vars)]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(f"{model.rsquared:.3f}")
new_data = pd.DataFrame({'area' : [100], 'height' : [10], 'wall' : [1]})
new_data = new_data[list(sig_vars)]
new_data = sm.add_constant(new_data, has_constant = 'add')
predict = model.predict(new_data)

import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(666)

n = 220

df = pd.DataFrame({
    'age': np.random.randint(20, 60, n),
    'income': np.random.normal(3500, 800, n),
    'overtime': np.random.choice([0, 1, 2], n)
})

z = (
    -3
    + 0.035 * df['age']
    - 0.00025 * df['income']
    + 0.8 * df['overtime']
    + np.random.normal(0, 1, n)
)

prob = 1 / (1 + np.exp(-z))
df['attrition'] = (prob >= 0.5).astype(int)

print(df.head())
print(df['attrition'].value_counts())
# attrition을 종속변수로 하고 나머지 변수를 독립변수로 하는 모형을 적합하시오.
# 유의확률이 0.05 미만인 변수의 개수를 구하시오.
# 단, 절편은 제외한다.
# age가 1 증가할 때의 오즈비를 구하시오.
# 소수 셋째 자리까지 출력하시오.
# age=30, income=3000, overtime=2인 데이터의 attrition=1 예측확률을 구하시오.
# 소수 셋째 자리까지 출력하시오.

X = df.drop(columns = 'attrition')
y = df['attrition']
X = sm.add_constant(X)

model = sm.Logit(y, X).fit()

cnt = (model.pvalues.drop('const') < 0.05).sum()
print(cnt)

new_data = pd.DataFrame({'age' : [30], 'income' : [3000], 'overtime' : [2]})
new_data = sm.add_constant(new_data, has_constant = 'add')
pred = model.predict(new_data)
print(pred.iloc[0])