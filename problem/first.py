import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
np.random.seed(42)

df = pd.DataFrame({
    'emp_id': range(1001, 1031),
    'dept': np.random.choice(['IT', 'HR', 'Sales', 'Finance'], 30),
    'gender': np.random.choice(['M', 'F'], 30),
    'age': np.random.randint(23, 55, 30),
    'salary': np.random.randint(2800, 7600, 30),
    'score': np.random.normal(75, 10, 30).round(1),
    'years': np.random.randint(1, 15, 30)
})

df.loc[[3, 8, 15], 'score'] = np.nan
df.loc[[5, 21], 'salary'] = np.nan

print(df.head())
# 1-1 score 결측치를 score의 중앙값으로 대체한 뒤, score 평균을 소수 둘째 자리까지 출력하시오.
df['score'] = df['score'].fillna(df['score'].median())
mean = df['score'].mean()
print(f'{mean: .2f}')

# 1-2 부서별 평균 salary를 구하고, 평균 salary가 가장 높은 부서명을 출력하시오.
print(df.groupby('dept')['salary'].mean().idxmax())
# 1-3 age가 35 이상이고 score가 80 이상인 직원 수를 출력하시오.
# print(df[(df['age'] >= 35) & (df['score'] >= 80)].value_counts().sum())
print(((df['age'] >= 35) & (df['score'] >= 80)).sum())
# 1-4 salary를 기준으로 내림차순 정렬했을 때 상위 5명의 salary 평균을 출력하시오.
print(df.sort_values('salary', ascending= False)['salary'].head().mean())
# 1-5 dept와 gender별 평균 score를 구하시오.
print(df.groupby(['dept','gender'])['score'].mean())
# 1-6 years가 10 이상이면 'senior', 아니면 'junior'인 새 컬럼 level을 만들고, level별 평균 salary를 출력하시오.
df['level'] = np.where(df['years'] >= 10, 'senior', 'junior')
print(df.groupby('level')['salary'].mean())
# 1-7 emp_id는 모델링에 넣어야 할 변수인지 판단하시오. 판단 근거로 nunique() 결과를 출력하시오.
print(df.shape)
print(df['emp_id'].nunique())

# 세트 2. 작업형 2: 회귀 문제
import pandas as pd
import numpy as np

np.random.seed(7)

n_train = 300
n_test = 120

train = pd.DataFrame({
    'house_id': range(10000, 10000 + n_train),
    'area': np.random.normal(85, 20, n_train).round(1),
    'rooms': np.random.randint(1, 6, n_train),
    'age': np.random.randint(0, 35, n_train),
    'district': np.random.choice(['A', 'B', 'C', 'D'], n_train),
    'near_subway': np.random.choice(['yes', 'no'], n_train)
})

test = pd.DataFrame({
    'house_id': range(20000, 20000 + n_test),
    'area': np.random.normal(85, 20, n_test).round(1),
    'rooms': np.random.randint(1, 6, n_test),
    'age': np.random.randint(0, 35, n_test),
    'district': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_test),
    'near_subway': np.random.choice(['yes', 'no'], n_test)
})

district_effect = {'A': 40, 'B': 20, 'C': 0, 'D': -15}
subway_effect = {'yes': 25, 'no': 0}

train['price'] = (
    train['area'] * 3.2
    + train['rooms'] * 12
    - train['age'] * 1.5
    + train['district'].map(district_effect)
    + train['near_subway'].map(subway_effect)
    + np.random.normal(0, 20, n_train)
).round(1)

train.loc[[4, 19, 55], 'area'] = np.nan
train.loc[[10, 88], 'district'] = np.nan
test.loc[[3, 30], 'area'] = np.nan
test.loc[[9], 'district'] = np.nan

print(train.head())
print(test.head())

# 2-1 price를 예측하는 회귀 모델을 만드시오.
# 조건:
# house_id는 제거할지 판단하시오.
# 숫자형 결측치는 평균으로 대체
# 범주형 결측치는 최빈값으로 대체
# 범주형 변수는 원핫인코딩
# 모델은 RandomForestRegressor(random_state=42) 사용
# train_test_split은 test_size=0.3, random_state=42
# 검증 데이터 RMSE를 출력하시오.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
print(train.shape)
print(train['house_id'].nunique())
X = train.drop(['house_id', 'price'], axis = 1)
y  = train['price']
test_X = test.drop(['house_id', 'price'], axis = 1, errors='ignore')

num_columns = X.select_dtypes('number').columns.tolist()
cat_columns = X.select_dtypes('object').columns.tolist()

num_preprocess = make_pipeline(
    SimpleImputer(strategy = 'mean')
)
cat_preprocess = make_pipeline(
    SimpleImputer(strategy = 'most_frequent'),
    OneHotEncoder(handle_unknown ='ignore', sparse_output = False)
)

preprocess = ColumnTransformer([
    ('num', num_preprocess, num_columns),
    ('cat', cat_preprocess, cat_columns)
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('regressor', RandomForestRegressor(random_state = 42))
])
from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42
)
pipe.fit(train_X, train_y)
price_pred = pipe.predict(valid_X)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(valid_y, price_pred)))

# 2-2 검증이 끝난 뒤 전체 train 데이터로 다시 학습하고, test 데이터의 price 예측값을 result_reg.csv로 저장하시오.
# 제출 파일 형식:
# pred
# 123.4
# 156.7

pipe.fit(X,y)
pred = pipe.predict(test_X)
pred = pd.DataFrame(pred, columns = ['pred'])
pred.to_csv('pred.csv', index = False)

# 2-3 house_id를 포함했을 때와 제거했을 때 RMSE를 각각 비교하시오. 어느 쪽이 더 적절한지 설명하시오.
exclude = np.sqrt(mean_squared_error(valid_y, price_pred))

X2 = train.drop(['price'], axis = 1)
y2 = train['price']
num_columns = X2.select_dtypes('number').columns.tolist()
cat_columns = X2.select_dtypes('object').columns.tolist()
preprocess = ColumnTransformer([
    ('num', num_preprocess, num_columns),
    ('cat', cat_preprocess, cat_columns)
])
pipe = Pipeline([
    ('preprocess', preprocess),
    ('regressor', RandomForestRegressor(random_state = 42))
])
train_X2, valid_X2, train_y2, valid_y2 = train_test_split(
    X2,
    y2,
    test_size = 0.3,
    random_state = 42
)
pipe.fit(train_X2, train_y2)
pred2 = pipe.predict(valid_X2)
include = np.sqrt(mean_squared_error(valid_y2, pred2))
print(f'제외한 RMSE: {exclude} \n 포함한 RMSE: {include}')
print('제외한 것이 약 1.4정도 낮으므로 제외한 것을 채택한다')
  
    
# 세트 3. 작업형 2: 분류 문제
import pandas as pd
import numpy as np

np.random.seed(11)

n_train = 400
n_test = 150

train = pd.DataFrame({
    'customer_id': range(5000, 5000 + n_train),
    'age': np.random.randint(18, 70, n_train),
    'income': np.random.normal(4200, 1200, n_train).round(1),
    'visit_count': np.random.poisson(5, n_train),
    'grade': np.random.choice(['silver', 'gold', 'vip'], n_train),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Daejeon'], n_train)
})

test = pd.DataFrame({
    'customer_id': range(9000, 9000 + n_test),
    'age': np.random.randint(18, 70, n_test),
    'income': np.random.normal(4200, 1200, n_test).round(1),
    'visit_count': np.random.poisson(5, n_test),
    'grade': np.random.choice(['silver', 'gold', 'vip', 'diamond'], n_test),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Daejeon'], n_test)
})

score = (
    -0.03 * train['age']
    + 0.0006 * train['income']
    + 0.35 * train['visit_count']
    + train['grade'].map({'silver': -0.5, 'gold': 0.2, 'vip': 1.0})
    + np.random.normal(0, 1, n_train)
)

prob = 1 / (1 + np.exp(-score))
train['churn'] = (prob > np.quantile(prob, 0.55)).astype(int)

train.loc[[6, 14, 25], 'income'] = np.nan
train.loc[[8, 41], 'grade'] = np.nan
test.loc[[2, 17], 'income'] = np.nan
test.loc[[13], 'grade'] = np.nan

print(train.head())
print(test.head())
print(train['churn'].value_counts())

# 3-1 churn을 예측하는 분류 모델을 만드시오.
# 조건:
# customer_id 제거 여부 판단
# 숫자형 결측치는 평균 대체
# 범주형 결측치는 최빈값 대체
# 범주형은 원핫인코딩
# 모델은 RandomForestClassifier(random_state=42)
# train_test_split은 test_size=0.3, random_state=42, stratify=y
# 검증 데이터의 accuracy와 f1_macro를 출력하시오.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

print(train.shape)
print(train['customer_id'].nunique())
X = train.drop(['customer_id', 'churn'], axis = 1)
y = train['churn']
test_X = test.drop(['customer_id', 'churn'], errors='ignore', axis = 1)

num_columns = X.select_dtypes('number').columns.tolist()
cat_columns = X.select_dtypes('object').columns.tolist()

num_preprocess = make_pipeline(SimpleImputer(strategy = 'mean'))
cat_preprocess = make_pipeline(SimpleImputer(strategy = 'most_frequent'), OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))

preprocess = ColumnTransformer([
    ('num', num_preprocess, num_columns),
    ('cat', cat_preprocess, cat_columns)
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('classifier', RandomForestClassifier(random_state = 42))
])

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42,
    stratify = y
)
pipe.fit(train_X, train_y)
val_pred = pipe.predict(valid_X)
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
print(f1_score(valid_y, val_pred, average = 'macro'))
print(accuracy_score(valid_y, val_pred))
# 3-2 평가 기준이 F1-macro라고 할 때, test 데이터에 대한 예측값을 result_cls.csv로 저장하시오.
pipe.fit(X,y)
pred = pipe.predict(test_X)
pred = pd.DataFrame(pred, columns = ['pred'])
pred.to_csv('result_cls.csv', index = False)


# 3-3 평가 기준이 AUC라고 바뀌면 제출 파일을 어떻게 바꿔야 하는지 코드로 작성하시오.
# 힌트:
# predict()
# predict_proba()
# 둘 중 뭘 써야 하는지 구분.
pipe.fit(X,y)
pred = pipe.predict_proba(test_X)[:, 1]
pred = pd.DataFrame(pred, columns = ['pred'])
pred.to_csv('result_cls.csv', index = False)

# 세트 4. 작업형 3: 통계검정

# 문제 4-1. 일표본 t검정 

# 어떤 반의 평균 점수가 75점보다 높은지 검정하려고 한다.
scores = np.array([78, 82, 74, 80, 77, 85, 79, 81, 76, 83])
# 유의수준 0.05에서 검정하시오.
# 출력할 것:
# 검정통계량
# p-value
# 귀무가설 기각 여부

from scipy import stats
stat, pvalue = stats.ttest_1samp(score, popmean = 75, alternative = 'greater')
print(stat, pvalue)
print('pvalue가 0.05보다 낮기 때문에 귀무가설을 기각한다')

# 문제 4-2. 독립표본 t검정
# A반과 B반의 점수 차이가 있는지 검정하시오.

class_A = np.array([81, 79, 85, 88, 84, 82, 90])
class_B = np.array([76, 74, 78, 80, 77, 75, 79])

# 조건:
# 양측검정
# 등분산 가정
# 유의수준 0.05


stat, pvalue = stats.ttest_ind(class_A, class_B, equal_var = True, alternative= 'two-sided')
print(stat, pvalue)
print('pvalue가 0.05보다 낮기 때문에 귀무가설을 기각한다')

# 문제 4-3. 대응표본 t검정
# 같은 학생들의 학습 전후 점수이다. 학습 후 점수가 증가했는지 검정하시오.

before = np.array([65, 70, 72, 68, 75, 71, 69, 73])
after = np.array([70, 74, 75, 72, 78, 76, 73, 77])

# 조건:
# 단측검정
# 유의수준 0.05

stat, pvalue = stats.ttest_rel(after, before, alternative = 'greater')
print(stat, pvalue)
print('pvalue가 0.05보다 낮기 때문에 귀무가설을 기각한다')

# 문제 4-4. 카이제곱 독립성 검정
# 성별과 구매 여부가 독립인지 검정하시오.

table = np.array([
    [30, 20],
    [18, 32]
])

# 행은 남/여, 열은 구매/미구매라고 하자.
# 출력할 것:
# chi-square statistic
# p-value
# 자유도
# 기대빈도
# 결론

chi2, p, df, expected = stats.chi2_contingency(table)
print(f'chi2: {chi2}')
print(f'pvalue: {p}')
print(f'df: {df}')
print(f'expected: \n{expected}')
print('pvalue가 0.05보다 낮기 떄문에 귀무가설을 기각한다')

# 문제 4-5. 카이제곱 적합도 검정
# 한 주사위를 60번 던진 결과가 다음과 같다.

observed = np.array([8, 12, 9, 11, 10, 10])
expected = np.repeat(10, 6)

# 주사위가 공정하다고 볼 수 있는지 유의수준 0.05에서 검정하시오.

stat, p = stats.chisquare(observed, f_exp = expected)
print(stat, p)
print('pvalue가 0.05보다 높기 때문에 귀무가설을 채택한다')
# 문제 4-6. ANOVA

# 세 가지 공부법에 따른 점수 차이가 있는지 검정하시오.

method_A = np.array([75, 78, 80, 77, 79])
method_B = np.array([82, 85, 84, 83, 86])
method_C = np.array([70, 72, 71, 69, 73])

# 조건:
# 일원분산분석
# 유의수준 0.05
# 귀무가설과 대립가설을 말로 쓰기

stat, p = stats.f_oneway(method_A, method_B, method_C)

print(stat, p)
print('귀무가설: 점수의 차이가 없다, 대립가설: 점수의 차이가 있다')
print('pvalue가 0.05보다 작으므로 귀무가설을 기각한다: 점수의 차이가 있다')

# 세트 5. OLS / Logit 문제

# 5-A. 선형회귀 문제

import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(21)

n = 120
df = pd.DataFrame({
    'study_time': np.random.normal(5, 2, n),
    'sleep_time': np.random.normal(7, 1.2, n),
    'attendance': np.random.normal(85, 8, n),
    'game_time': np.random.normal(2, 1, n)
})

df['score'] = (
    10
    + 4.5 * df['study_time']
    + 1.8 * df['sleep_time']
    + 0.35 * df['attendance']
    - 3.2 * df['game_time']
    + np.random.normal(0, 5, n)
)

print(df.head())
# 문제 5-A-1 score와 가장 상관관계가 큰 변수를 찾고, 해당 상관계수를 출력하시오.

# 주의:
# 변수 선택은 절댓값 기준
# 출력 상관계수는 원래 부호 유지

corr = df.corr()['score'].drop('score')
max_var = corr.abs().idxmax()
max_corr = corr[max_var]
print(max_var, max_corr)

# 문제 5-A-2

# score를 종속변수로 하고 나머지 네 변수를 독립변수로 하는 다중선형회귀모형을 적합하시오.

# 출력:
# model.summary()
# 결정계수 R²
# 수정 결정계수 Adj. R²

import statsmodels.api as sm
X = df.drop(columns =  'score')
y = df['score']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

print(model.rsquared)
print(model.rsquared_adj)

# 문제 5-A-3
# p-value가 가장 큰 변수를 찾고, 해당 p-value를 출력하시오.

# 주의:
# const는 제외

pvalues = model.pvalues.drop('const')
max_p_var = pvalues.idxmax()
max_p = pvalues[max_p_var]
print(max_p_var, max_p)
# 문제 5-A-4

# 유의수준 0.05에서 통계적으로 유의한 변수의 개수를 구하시오.
# 주의:
# const 제외

cnt = (pvalues < 0.05).sum()
print(cnt)

# 문제 5-A-5

# 다음 조건에서 예측 점수를 계산하시오.
study_time = 6
sleep_time = 7.5
attendance = 90
game_time = 1.5

new_data = pd.DataFrame({
    'study_time' : [6],
    'sleep_time' : [7.5],
    'attendance' : [90],
    'game_time' : [1.5]
})
new_data = sm.add_constant(new_data, has_constant = 'add')
pred = model.predict(new_data)

print(pred.iloc[0])
# 5-B. 로지스틱 회귀 문제
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(33)

n = 250
df = pd.DataFrame({
    'hours': np.random.normal(4, 1.5, n),
    'attendance': np.random.normal(80, 10, n),
    'previous_score': np.random.normal(70, 12, n),
    'late_count': np.random.poisson(2, n)
})

z = (
    -6
    + 0.7 * df['hours']
    + 0.04 * df['attendance']
    + 0.03 * df['previous_score']
    - 0.5 * df['late_count']
)

prob = 1 / (1 + np.exp(-z))
df['pass'] = (prob > np.quantile(prob, 0.45)).astype(int)

print(df.head())
print(df['pass'].value_counts())
# 문제 5-B-1

# pass를 종속변수로 하고 hours만 독립변수로 하는 로지스틱 회귀모형을 적합하시오.
# hours의 오즈비를 계산하시오.

X = df[['hours']]
y = df['pass']

X = sm.add_constant(X)

model = sm.Logit(y, X).fit()

odds_ratio = np.exp(model.params['hours'])
print(odds_ratio)

# 문제 5-B-2

# pass를 종속변수로 하고 모든 변수를 독립변수로 하는 로지스틱 회귀모형을 적합하시오.
# Residual deviance를 계산하시오.

# 공식:
# -2 * model.llf

X = df.drop(columns = 'pass')
y = df['pass']

X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
resid_dev = -2 * model.llf
print(resid_dev)

# 문제 5-B-3
# 모든 변수를 사용한 로지스틱 회귀모형에서 p-value가 0.05 이상인 변수 개수를 구하시오.

# 주의:
# const 포함할지 문제에서 따로 말 안 했으면 제외하고 계산

pvalue = model.pvalues.drop('const')
cnt = (pvalue >= 0.05).sum()
print(cnt)

# 문제 5-B-4
# 유의한 변수만 사용해서 다시 로지스틱 회귀모형을 적합하고, 유의한 변수들의 회귀계수 평균을 구하시오.

# 주의:
# 평균 계산에서 const 제외

sig_vars = model.pvalues.drop('const')
sig_vars = sig_vars[sig_vars < 0.05].index

X_sig =  df[list(sig_vars)]
X_sig = sm.add_constant(X_sig)

model = sm.Logit(y, X_sig).fit()
coef_mean = model.params.drop('const').mean()
print(coef_mean)

# 문제 5-B-5
# 데이터를 학습/평가로 나누시오.

# 조건:
# test_size=80
# random_state=42
# 모든 변수를 사용한 로지스틱 회귀모형을 학습한 뒤, 평가 데이터의 오분류율을 계산하시오.

# 기준:
# 예측확률 >= 0.5 이면 1

train_df, test_df = train_test_split(
    df,
    test_size = 80,
    random_state = 42
)

train_X = train_df.drop(columns = 'pass')
train_y = train_df['pass']
test_X = test_df.drop(columns = 'pass')
test_y = test_df['pass']

train_X = sm.add_constant(train_X)
test_X = sm.add_constant(test_X, has_constant = 'add')

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 1000)
model.fit(train_X, train_y)
prob = model.predict_proba(test_X)[:, 1]
pred = (prob >= 0.5).astype(int)
error_rate = 1 - accuracy_score(test_y, pred)
print(error_rate)

# 문제 5-B-6

# 평가 데이터에서 AUC를 계산하시오.

# 주의:
# AUC는 class label이 아니라 확률값으로 계산

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_y, prob)
print(auc)