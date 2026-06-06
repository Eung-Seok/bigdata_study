# 작업형 1
import pandas as pd
import numpy as np

np.random.seed(123)

df = pd.DataFrame({
    'user_id': range(2001, 2051),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Incheon'], 50),
    'membership': np.random.choice(['basic', 'plus', 'premium'], 50),
    'age': np.random.randint(19, 65, 50),
    'purchase_count': np.random.randint(0, 20, 50),
    'amount': np.random.normal(120000, 35000, 50).round(0),
    'satisfaction': np.random.normal(4.0, 0.7, 50).round(1)
})

df.loc[[4, 17, 31], 'amount'] = np.nan
df.loc[[8, 22], 'satisfaction'] = np.nan

print(df.head())
print(df.info())
# 1-1
# amount 결측치를 amount의 평균값으로 대체한 뒤,
# membership별 평균 amount를 구하고 평균 금액이 가장 높은 membership을 출력하시오.

df['amount'] = df['amount'].fillna(df['amount'].mean())
print(df.groupby('membership')['amount'].mean().idxmax())

# 1-2
# satisfaction 결측치를 중앙값으로 대체한 뒤,
# age가 40 이상이고 satisfaction이 4.2 이상인 사람의 수를 출력하시오.
df['satisfaction'] = df['satisfaction'].fillna(df['satisfaction'].median())
print(((df['age'] >= 40) & (df['satisfaction'] >= 4.2)).sum())

# 1-3
# user_id가 모델링에 필요한 변수인지 판단하시오.

# 출력할 것:
# 전체 행 수
# user_id의 고유값 개수
# 제거 여부 판단 문장

print(df.shape)
print(df['user_id'].nunique())
print('데이터의 수와 user_id의 unique의 수가 동일하기 때문에 user_id를 식별자로써 간주할 수 있다. 따라서 모델링에서는 user_id를 제외한다')

# 작업형 2: 분류 문제
import pandas as pd
import numpy as np

np.random.seed(321)

n_train = 500
n_test = 180

train = pd.DataFrame({
    'app_id': range(10000, 10000 + n_train),
    'age': np.random.randint(20, 70, n_train),
    'income': np.random.normal(4500, 1300, n_train).round(1),
    'loan_amount': np.random.normal(2500, 800, n_train).round(1),
    'credit_score': np.random.normal(650, 70, n_train).round(1),
    'job_type': np.random.choice(['office', 'self', 'public', 'student'], n_train),
    'house_owner': np.random.choice(['yes', 'no'], n_train)
})

test = pd.DataFrame({
    'app_id': range(20000, 20000 + n_test),
    'age': np.random.randint(20, 70, n_test),
    'income': np.random.normal(4500, 1300, n_test).round(1),
    'loan_amount': np.random.normal(2500, 800, n_test).round(1),
    'credit_score': np.random.normal(650, 70, n_test).round(1),
    'job_type': np.random.choice(['office', 'self', 'public', 'student', 'freelancer'], n_test),
    'house_owner': np.random.choice(['yes', 'no'], n_test)
})

score = (
    0.002 * train['income']
    - 0.0025 * train['loan_amount']
    + 0.01 * train['credit_score']
    - 0.02 * train['age']
    + train['job_type'].map({'office': 0.3, 'self': -0.2, 'public': 0.5, 'student': -0.5})
    + train['house_owner'].map({'yes': 0.4, 'no': 0})
    + np.random.normal(0, 1.0, n_train)
)

prob = 1 / (1 + np.exp(-score))
train['approved'] = (prob > np.quantile(prob, 0.45)).astype(int)

train.loc[[3, 9, 44], 'income'] = np.nan
train.loc[[11, 87], 'job_type'] = np.nan
test.loc[[5, 33], 'income'] = np.nan
test.loc[[18], 'job_type'] = np.nan

print(train.head())
print(test.head())
print(train['approved'].value_counts())
# 2-1
# approved를 예측하는 분류 모델을 만드시오.

# 조건:
# app_id는 제거할지 판단하시오.
# 숫자형 결측치는 평균으로 대체
# 범주형 결측치는 최빈값으로 대체
# 범주형 변수는 원핫인코딩
# test 데이터에 train에 없던 범주가 있어도 에러가 나지 않게 처리
# 모델은 RandomForestClassifier(random_state=42)
# train_test_split은 test_size=0.3, random_state=42, stratify=y
# 검증 데이터의 accuracy, f1_macro, AUC를 출력하시오.

print(train.shape, train['app_id'].nunique())
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

X = train.drop(['app_id', 'approved'], axis = 1)
y = train['approved']

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
f1 = f1_score(valid_y, pred, average='macro')
accuracy = accuracy_score(valid_y, pred)
auc = roc_auc_score(valid_y, prob)

print(f'f1-macro: {f1}')
print(f'accuracy: {accuracy}')
print(f'AUC: {auc}')

# 2-2
# 평가 기준이 AUC라고 할 때,
# test 데이터의 예측 결과를 result_auc.csv로 저장하시오.

# 제출 컬럼명은 pred로 하시오.

# 주의:
# AUC 기준이면 class label이 아니라 확률값 제출

pipe.fit(X,y)
test_X = test.drop(['app_id', 'approved'], errors='ignore', axis = 1)
prob = pipe.predict_proba(test_X)[:, 1]
prob = pd.DataFrame(prob, columns = ['pred'])
prob.to_csv('result_auc.csv', index = False)
# 작업형 3-1: 통계검정
# 아래 데이터는 두 광고 방식 A, B를 본 고객들의 구매 금액이다.
import numpy as np
from scipy import stats

ad_A = np.array([42, 45, 39, 50, 47, 44, 46, 48, 43, 49])
ad_B = np.array([38, 40, 37, 41, 39, 36, 42, 40, 35, 39])
# 문제

# 광고 A의 평균 구매 금액이 광고 B보다 크다고 할 수 있는지 유의수준 0.05에서 검정하시오.

# 출력할 것:

# 1. 귀무가설과 대립가설
# 2. 사용할 검정명
# 3. 검정통계량
# 4. p-value
# 5. 결론

stat, pvalue = stats.ttest_ind(ad_A, ad_B,equal_var = True, alternative = 'greater')
print(pvalue)
print('pvalue가 0.05보다 낮기 때문에 귀무가설을 기각하고, A의 평균 구매 금액이 B보다 크다고 볼 수 있다')
# 작업형 3-2: 카이제곱 검정
# 아래 표는 지역별 상품 선호도 조사 결과이다.
import numpy as np
from scipy import stats

table = np.array([
    [30, 20, 10],
    [25, 25, 20],
    [15, 30, 25]
])

# 행은 지역 A, B, C, 열은 상품 X, Y, Z라고 하자.

# 문제

# 지역과 상품 선호도가 독립인지 유의수준 0.05에서 검정하시오.

# 출력할 것:

# 1. 귀무가설과 대립가설
# 2. chi-square statistic
# 3. p-value
# 4. 자유도
# 5. 기대빈도
# 6. 결론

chi2, pvalue, df, expected = stats.chi2_contingency(table)
print(chi2)
print(pvalue)
print(df)
print(expected)
print('pvalue가 0.05보다 작으므로 귀무가설을 기각한다: 독립이 아니다')

# 작업형 3-3: OLS 문제

# 아래 코드 실행.

import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(777)

n = 150

df = pd.DataFrame({
    'ad_budget': np.random.normal(100, 25, n),
    'price': np.random.normal(30, 5, n),
    'store_count': np.random.normal(20, 4, n),
    'competitor_price': np.random.normal(28, 6, n)
})

df['sales'] = (
    50
    + 2.8 * df['ad_budget']
    - 4.5 * df['price']
    + 6.0 * df['store_count']
    + 1.5 * df['competitor_price']
    + np.random.normal(0, 20, n)
)

print(df.head())
# 문제

# sales를 종속변수로 하고 나머지 변수를 독립변수로 하는 다중선형회귀모형을 적합하시오.

# 출력할 것:
# 1. model.summary()
# 2. R-squared
# 3. p-value가 가장 큰 변수명과 p-value
# 4. 유의수준 0.05에서 유의한 변수 개수
# 5. ad_budget=120, price=32, store_count=22, competitor_price=30일 때 예측 sales
# 주의:
# p-value 문제에서 const 제외
# 예측할 때 const 추가

X = df.drop(columns = 'sales')
y = df['sales']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
print(model.rsquared)
print(model.pvalues.drop('const').idxmax(), model.pvalues.drop('const').max())
print((model.pvalues.drop('const') <= 0.05).sum())
new_data = pd.DataFrame({
    'ad_budget' : [120],
    'price' : [32],
    'store_count' : [22],
    'competitor_price' : [30]
})
new_data = sm.add_constant(new_data, has_constant = 'add')
pred = model.predict(new_data)
print(pred.iloc[0])