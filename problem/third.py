# 작업형 1 대량 세트

import pandas as pd
import numpy as np

np.random.seed(2026)

df = pd.DataFrame({
    'order_id': range(10001, 10081),
    'customer_id': np.random.randint(2000, 2035, 80),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju'], 80),
    'channel': np.random.choice(['online', 'offline', 'app'], 80),
    'membership': np.random.choice(['basic', 'silver', 'gold', 'vip'], 80),
    'age': np.random.randint(18, 70, 80),
    'quantity': np.random.randint(1, 8, 80),
    'price': np.random.normal(50000, 12000, 80).round(0),
    'discount_rate': np.random.choice([0, 0.05, 0.1, 0.15, 0.2], 80),
    'satisfaction': np.random.normal(4.0, 0.8, 80).round(1)
})

df.loc[[3, 11, 24, 55], 'price'] = np.nan
df.loc[[5, 19, 41], 'satisfaction'] = np.nan
df.loc[[7, 30], 'membership'] = np.nan

print(df.head())
print(df.info())
# 1-1 price 결측치를 price의 중앙값으로 대체한 뒤, 전체 price 평균을 소수 둘째 자리까지 출력하시오.
df['price'] = df['price'].fillna(df['price'].median())
print(f'평균 소수 둘째: {df['price'].mean():.2f}')
# 1-2 membership 결측치를 최빈값으로 대체한 뒤, membership별 평균 satisfaction을 구하고 평균 만족도가 가장 높은 등급명을 출력하시오.
from sklearn.impute import SimpleImputer
df[['membership']] = SimpleImputer(strategy = 'most_frequent').fit_transform(df[['membership']])
print(df.groupby('membership')['satisfaction'].mean())
print(df.groupby('membership')['satisfaction'].mean().idxmax())
# 1-3 satisfaction 결측치를 중앙값으로 대체한 뒤, age >= 40이고 satisfaction >= 4.3인 행의 개수를 출력하시오.
df['satisfaction'] = df['satisfaction'].fillna(df['satisfaction'].median())
cnt = ((df['age'] >= 40) & (df['satisfaction'] >= 4.3)).sum()
print(cnt)
# 1-4 quantity * price * (1 - discount_rate)를 계산하여 final_amount 컬럼을 만드시오.
# final_amount가 가장 큰 상위 10개 주문의 평균 final_amount를 출력하시오.
df['final_amount'] = df['quantity'] * df['price'] * (1 - df['discount_rate'])
print(df['final_amount'].sort_values(ascending = False).head(10).mean())
# 1-5 region과 channel별 final_amount 평균을 구하시오.
print(df.groupby(['region', 'channel'])['final_amount'].mean())
# 1-6 age가 50 이상이면 'old', 30 이상 50 미만이면 'middle', 30 미만이면 'young'인 age_group 컬럼을 만드시오.
# age_group별 평균 satisfaction을 출력하시오.
df['age_group'] = np.where(df['age'] >= 50, 'old', np.where(df['age'] >= 30, 'middle', 'young'))
print(df.groupby('age_group')['satisfaction'].mean())
# 1-7 order_id와 customer_id 중 모델링에서 제거해야 할 가능성이 더 높은 컬럼을 판단하시오.
# 출력할 것:
# 전체 행 수
# order_id 고유값 개수
# customer_id 고유값 개수
# 제거할 컬럼명
# 판단 이유
print(df.shape)
print(df['order_id'].nunique())
print(df['customer_id'].nunique())
print('order_id는 전체 행 수와 같고, customer_id는 절반이하이다. 따라서 customer_id보다는 식별자 ID열로 판단할 수 있는 order_id는  제거해야한다.')
# 1-8 region별 주문 수를 구하고, 주문 수가 가장 많은 지역명을 출력하시오.
print(df.groupby('region')['quantity'].sum().idxmax())

# 작업형 2-A: 회귀 문제

import pandas as pd
import numpy as np

np.random.seed(77)

n_train = 450
n_test = 180

train = pd.DataFrame({
    'car_id': range(100000, 100000 + n_train),
    'brand': np.random.choice(['A', 'B', 'C', 'D'], n_train),
    'fuel': np.random.choice(['gasoline', 'diesel', 'hybrid'], n_train),
    'year': np.random.randint(2010, 2024, n_train),
    'mileage': np.random.normal(80000, 30000, n_train).round(0),
    'engine_size': np.random.normal(2.0, 0.5, n_train).round(2),
    'accident_count': np.random.poisson(0.4, n_train)
})

test = pd.DataFrame({
    'car_id': range(200000, 200000 + n_test),
    'brand': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_test),
    'fuel': np.random.choice(['gasoline', 'diesel', 'hybrid', 'electric'], n_test),
    'year': np.random.randint(2010, 2024, n_test),
    'mileage': np.random.normal(80000, 30000, n_test).round(0),
    'engine_size': np.random.normal(2.0, 0.5, n_test).round(2),
    'accident_count': np.random.poisson(0.4, n_test)
})

brand_effect = {'A': 300, 'B': 150, 'C': 0, 'D': -100}
fuel_effect = {'gasoline': 0, 'diesel': -80, 'hybrid': 120}

train['price'] = (
    2000
    + (train['year'] - 2010) * 120
    - train['mileage'] * 0.008
    + train['engine_size'] * 350
    - train['accident_count'] * 180
    + train['brand'].map(brand_effect)
    + train['fuel'].map(fuel_effect)
    + np.random.normal(0, 250, n_train)
).round(1)

train.loc[[10, 22, 48, 100], 'mileage'] = np.nan
train.loc[[13, 57], 'brand'] = np.nan
test.loc[[5, 35], 'mileage'] = np.nan
test.loc[[9], 'brand'] = np.nan

print(train.head())
print(test.head())
print(train.info())
print(test.info())

# 2-A-1 price를 예측하는 회귀 모델을 만드시오.

# 조건:
# car_id는 제거 여부를 판단하시오.
# 숫자형 결측치는 평균으로 대체
# 범주형 결측치는 최빈값으로 대체
# 범주형 변수는 원핫인코딩
# test에 train에 없던 범주가 있어도 에러가 나지 않게 처리
# 모델은 RandomForestRegressor(random_state=42)
# train_test_split은 test_size=0.3, random_state=42
# 검증 데이터 RMSE를 출력하시오.

print(train.shape, train['car_id'].nunique())
X = train.drop(['car_id', 'price'], axis = 1)
y = train['price']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
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

train_X, valid_X, train_y, valid_y = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42
)

from sklearn.metrics import mean_squared_error
pipe.fit(train_X, train_y)
val_pred = pipe.predict(valid_X)
print('RMSE: ', np.sqrt(mean_squared_error(valid_y, val_pred)))



# 2-A-2 검증 후 전체 train 데이터로 다시 학습하고, test 데이터의 예측값을 result_reg2.csv로 저장하시오.
# 컬럼명: pred
test_X = test.drop(['car_id', 'price'], axis = 1, errors = 'ignore')
pipe.fit(X,y)
pred = pipe.predict(test_X)
pred = pd.DataFrame(pred, columns = ['pred'])
pred.to_csv('result_reg2.csv', index=  False)



# 2-A-3 car_id를 포함한 경우와 제외한 경우의 RMSE를 비교하고, 어떤 선택이 더 적절한지 설명하시오.

X2 = train.drop('price', axis = 1)
y2 = train['price']
num_columns2 = X2.select_dtypes('number').columns.tolist()
cat_columns2 = X2.select_dtypes('object').columns.tolist()

preprocess2 = ColumnTransformer([
    ('num', num_preprocess, num_columns2),
    ('cat', cat_preprocess, cat_columns2)
])

pipe2 = Pipeline([
    ('preprocess', preprocess2),
    ('regressor', RandomForestRegressor(random_state = 42))
])

train_X2, valid_X2, train_y2, valid_y2 = train_test_split(
    X2,
    y2,
    test_size = 0.3,
    random_state = 42
)
pipe2.fit(train_X2, train_y2)
pred2 = pipe2.predict(valid_X2)
print('RMSE: ', np.sqrt(mean_squared_error(valid_y, val_pred)))
print('RMSE2: ', np.sqrt(mean_squared_error(valid_y2, pred2)))

print('car_id를 뺀 모델이 RMSE가 더 크므로 성능이 안좋다고 볼 수 있다. 따라서 car_id를 제외한 모델을 채택한다')
# 작업형 2-B: 분류 문제

import pandas as pd
import numpy as np

np.random.seed(88)

n_train = 520
n_test = 200

train = pd.DataFrame({
    'member_no': range(30000, 30000 + n_train),
    'age': np.random.randint(18, 75, n_train),
    'monthly_fee': np.random.normal(50000, 15000, n_train).round(0),
    'login_count': np.random.poisson(12, n_train),
    'watch_hours': np.random.normal(30, 12, n_train).round(1),
    'plan': np.random.choice(['basic', 'standard', 'premium'], n_train),
    'device': np.random.choice(['mobile', 'pc', 'tv'], n_train)
})

test = pd.DataFrame({
    'member_no': range(40000, 40000 + n_test),
    'age': np.random.randint(18, 75, n_test),
    'monthly_fee': np.random.normal(50000, 15000, n_test).round(0),
    'login_count': np.random.poisson(12, n_test),
    'watch_hours': np.random.normal(30, 12, n_test).round(1),
    'plan': np.random.choice(['basic', 'standard', 'premium', 'family'], n_test),
    'device': np.random.choice(['mobile', 'pc', 'tv', 'tablet'], n_test)
})

score = (
    -0.03 * train['age']
    -0.00002 * train['monthly_fee']
    -0.15 * train['login_count']
    -0.08 * train['watch_hours']
    + train['plan'].map({'basic': 0.6, 'standard': 0.2, 'premium': -0.4})
    + train['device'].map({'mobile': 0.2, 'pc': 0.0, 'tv': -0.2})
    + np.random.normal(0, 1, n_train)
)

prob = 1 / (1 + np.exp(-score))
train['churn'] = (prob > np.quantile(prob, 0.60)).astype(int)

train.loc[[4, 40, 99], 'monthly_fee'] = np.nan
train.loc[[12, 77], 'plan'] = np.nan
test.loc[[6, 60], 'monthly_fee'] = np.nan
test.loc[[15], 'plan'] = np.nan

print(train.head())
print(test.head())
print(train['churn'].value_counts())

# 2-B-1 churn을 예측하는 분류 모델을 만드시오.

# 조건:
# member_no는 제거 여부를 판단하시오.
# 숫자형 결측치는 평균으로 대체
# 범주형 결측치는 최빈값으로 대체
# 범주형 변수는 원핫인코딩
# test에 train에 없던 범주가 있어도 에러가 나지 않게 처리
# 모델은 RandomForestClassifier(random_state=42)
# train_test_split은 test_size=0.3, random_state=42, stratify=y
# 검증 데이터의 accuracy, f1_macro, AUC를 출력하시오.

print(train.shape, train['member_no'].nunique())

X = train.drop(['member_no', 'churn'], axis = 1)
y = train['churn']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
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
    ('classifier', RandomForestClassifier(random_state = 42))
])

train_X, valid_X, train_y, valid_y = train_test_split(
    X,
    y,
    test_size = 0.3,
    random_state = 42,
    stratify = y
)

pipe.fit(train_X, train_y)
pred = pipe.predict(valid_X)
prob = pipe.predict_proba(valid_X)[:, 1]

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

print(f'f1-macro: {f1_score(valid_y, pred, average = 'macro')}')
print(f'accuracy: {accuracy_score(valid_y, pred)}')
print(f'AUC: {roc_auc_score(valid_y, prob)}')

# 2-B-2 평가 기준이 F1-macro일 때 test 예측값을 result_cls2.csv로 저장하시오.

pipe.fit(X,y)
test_X = test.drop(['member_no', 'churn'], errors = 'ignore', axis = 1)
pred = pipe.predict(test_X)
pred = pd.DataFrame(pred, columns = ['pred'])
pred.to_csv('result_cls2.csv', index = False)

# 2-B-3 평가 기준이 AUC일 때 test 예측값을 result_auc2.csv로 저장하시오.
prob = pipe.predict_proba(test_X)[:, 1]
prob = pd.DataFrame(prob, columns = ['pred'])
prob.to_csv('result_auc2.csv', index = False)

# 작업형 3 대량 세트
# 3-1. 일표본 t검정
import numpy as np
from scipy import stats

delivery_time = np.array([28, 31, 29, 35, 30, 32, 27, 34, 33, 29, 31, 30])

# 한 배달 서비스의 평균 배달 시간이 30분보다 길다고 할 수 있는지 유의수준 0.05에서 검정하시오.
# 출력:
# 검정명
# 귀무가설 / 대립가설
# 검정통계량
# p-value
# 결론

stat, pvalue  = stats.ttest_1samp(delivery_time, popmean = 30, alternative = 'greater')
print(stat, pvalue)
print('귀무가설: 30분보다 작다. 대립가설: 30분보다 크다')
print('pvalue가 0.05보다 크기 때문에 귀무가설을 기각할 수 없다: 30분보다 작다')
# 3-2. 독립표본 t검정
group_A = np.array([85, 88, 90, 86, 87, 91, 89, 92])
group_B = np.array([80, 82, 79, 83, 81, 84, 78, 82])

# 두 교육 방식 A, B의 평균 점수에 차이가 있는지 유의수준 0.05에서 검정하시오.

# 조건:
# 등분산 가정
# 양측검정

stat, pvalue = stats.ttest_ind(group_A, group_B, equal_var = True, alternative = 'two-sided')
print(pvalue)
print('귀무가설: 평균점수에 차이가 없다. 대립가설: 평균점수에 차이가 있다')
print('pvalue가 0.05보다 작기 때문에 귀무가설을 기각한다: 평균점수에 차이가 있다')

# 3-3. 대응표본 t검정
before = np.array([72, 75, 70, 68, 74, 71, 73, 69, 76])
after = np.array([75, 78, 74, 70, 77, 73, 76, 72, 79])

# 같은 학생들의 수업 전후 점수이다.
# 수업 후 점수가 증가했는지 유의수준 0.05에서 검정하시오.

stat, pvalue = stats.ttest_rel(after, before, alternative = 'greater')
print(stat, pvalue)
print('귀무가설: 수업 후 점수가 증가하지 않았다. 대립가설: 수업 후 점수가 증가했다.')
print('pvalue가 0.05보다 작기 때문에 귀무가설을 기각한다. 수업 후 점수가 증가했다')
# 3-4. 카이제곱 독립성 검정
table = np.array([
    [40, 25],
    [30, 35],
    [20, 45]
])

# 행은 연령대 20대, 30대, 40대, 열은 구매 여부 구매, 미구매라고 하자.
# 연령대와 구매 여부가 독립인지 유의수준 0.05에서 검정하시오.

# 출력:
# chi-square statistic
# p-value
# df
# expected
# 결론

chi2, pvalue, df, expected = stats.chi2_contingency(table)
print(chi2)
print(pvalue)
print(df)
print(expected)
print('귀무가설: 독립이다. 대립가설: 독립이 아니다')
print('pvalue가 0.05보다 작기 때문에 귀무가설을 기각한다. : 독립이 아니다.')

# 3-5. 카이제곱 적합도 검정
observed = np.array([18, 22, 20, 15, 25])
expected = np.repeat(20, 5)

# 5개 지점의 방문자 수가 동일한 비율이라고 볼 수 있는지 유의수준 0.05에서 검정하시오.
stat, pvalue = stats.chisquare(observed, f_exp = expected)
print(stat, pvalue)

# 3-6. ANOVA
method_A = np.array([65, 68, 70, 67, 69])
method_B = np.array([72, 75, 74, 73, 76])
method_C = np.array([66, 64, 65, 67, 63])
method_D = np.array([80, 82, 79, 81, 83])

# 네 가지 교육 방법에 따른 평균 점수 차이가 있는지 유의수준 0.05에서 검정하시오.

# 출력:
# 검정명
# 귀무가설 / 대립가설
# F-statistic
# p-value
# 결론


# 3-7. 정규성 검정
sample = np.array([12.1, 11.8, 12.5, 13.0, 11.7, 12.2, 12.4, 11.9, 12.6, 12.0])

# 위 데이터가 정규분포를 따른다고 볼 수 있는지 Shapiro-Wilk 검정을 수행하시오.

# 출력:
# 검정통계량
# p-value
# 결론


# 3-8. 등분산성 검정
class_1 = np.array([70, 72, 71, 73, 74, 72])
class_2 = np.array([80, 85, 78, 90, 88, 84])

# 두 집단의 분산이 같다고 볼 수 있는지 Levene 검정을 수행하시오.

# 3-9. OLS 문제

import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(909)

n = 180

df = pd.DataFrame({
    'temperature': np.random.normal(22, 5, n),
    'advertising': np.random.normal(100, 30, n),
    'holiday': np.random.choice([0, 1], n),
    'competitor_sales': np.random.normal(300, 70, n)
})

df['sales'] = (
    200
    + 8.5 * df['temperature']
    + 3.2 * df['advertising']
    + 45 * df['holiday']
    - 0.4 * df['competitor_sales']
    + np.random.normal(0, 30, n)
)

print(df.head())
# 문제: sales를 종속변수로 하고 나머지 변수를 독립변수로 하는 다중선형회귀모형을 적합하시오.

# 출력:
# 1. model.summary()
# 2. R-squared
# 3. 수정 R-squared
# 4. p-value가 가장 큰 변수명과 p-value
# 5. 유의수준 0.05에서 유의한 변수 개수
# 6. 회귀계수가 가장 큰 변수명과 회귀계수
# 7. temperature=25, advertising=120, holiday=1, competitor_sales=280일 때 예측 sales

# 주의:
# p-value 문제에서 const 제외
# 회귀계수 가장 큰 변수 문제에서도 const 제외
# 예측할 때 const 추가
X = df.drop(columns = 'sales')
y = df['sales']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
print(model.rsquared)
print(model.rsquared_adj)
print(model.pvalues.drop('const').idxmax(), model.pvalues.drop('const').max())
print((model.pvalues.drop('const') < 0.05).sum())
print(model.params.drop('const').idxmax(), model.params.drop('const').max())
new_data = pd.DataFrame({'temperature' : [25], 'advertising': [120], 'holiday' : [1], 'competitor_sales' : [280]})
new_data = sm.add_constant(new_data, has_constant = 'add')
print(model.predict(new_data).iloc[0])
# 3-10. Logit 문제

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(1004)

n = 300

df = pd.DataFrame({
    'study_hours': np.random.normal(5, 1.5, n),
    'attendance': np.random.normal(80, 8, n),
    'assignment_score': np.random.normal(70, 10, n),
    'absence_count': np.random.poisson(2, n)
})

z = (
    -4
    + 0.5 * df['study_hours']
    + 0.03 * df['attendance']
    + 0.025 * df['assignment_score']
    - 0.35 * df['absence_count']
    + np.random.normal(0, 1.2, n)
)

prob = 1 / (1 + np.exp(-z))
df['pass'] = (prob > 0.55).astype(int)

print(df.head())
print(df['pass'].value_counts())

# 3-10-1 pass를 종속변수로 하고 study_hours만 독립변수로 하는 로지스틱 회귀모형을 적합하시오.
# study_hours의 오즈비를 출력하시오.
X = df.drop(['attendance', 'assignment_score', 'absence_count', 'pass'], axis = 1)
y = df['pass']
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
print(np.exp(model.params[1]))
# 3-10-2 pass를 종속변수로 하고 모든 변수를 독립변수로 하는 로지스틱 회귀모형을 적합하시오.
# Residual deviance를 계산하시오.
X = df.drop(columns = 'pass')
y = df['pass']
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
print(-2 * model.llf)
# 3-10-3 모든 변수를 사용한 로지스틱 회귀모형에서 p-value가 0.05 이상인 변수 개수를 구하시오.
cnt = (model.pvalues.drop('const') >= 0.05).sum()
print(cnt)

# 3-10-4 유의한 변수만 사용해서 다시 로지스틱 회귀모형을 적합하고, 유의한 변수들의 회귀계수 평균을 구하시오.
sig_var = (model.pvalues.drop('const') < 0.05)
X = df.drop(['attendance', 'pass'], axis = 1)
y = df['pass']
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
print(model.params.drop('const').mean())

# 주의: 평균 계산에서 const 제외

# 3-10-5 데이터를 학습/평가로 나누시오.

# 조건:
# test_size=90
# random_state=42

# 모든 변수를 사용하여 로지스틱 회귀모형을 학습한 뒤, 평가 데이터의 오분류율을 계산하시오.

# 기준: 예측확률 >= 0.5이면 1

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    test_size = 90,
    random_state = 42
)
model = sm.Logit(train_y, train_X).fit()
prob = model.predict(test_X)
pred = (prob >= 0.5).astype(int)
from sklearn.metrics import accuracy_score
error_rate = 1 - accuracy_score(test_y, pred)
print(error_rate)
# 3-10-6 3-10-5의 평가 데이터에서 AUC를 계산하시오.

# 주의: AUC는 class label이 아니라 확률값 사용

from sklearn.metrics import roc_auc_score
print(roc_auc_score(test_y, prob))