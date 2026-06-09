import pandas as pd
import numpy as np

np.random.seed(801)

df = pd.DataFrame({
    'emp_id': range(1001, 1121),
    'dept': np.random.choice(['A', 'B', 'C', 'D'], 120),
    'gender': np.random.choice(['M', 'F'], 120),
    'join_date': pd.date_range('2017-01-01', periods=120, freq='23D'),
    'age': np.random.randint(24, 61, 120),
    'score': np.random.normal(78, 11, 120).round(1),
    'salary': np.random.normal(4800, 850, 120).round(0),
    'work_year': np.random.normal(6, 2, 120).round(1),
    'comment': np.random.choice([
        'fast good response',
        'late delay issue',
        'normal service',
        'excellent fast support',
        'bad complaint delay'
    ], 120)
})

df.loc[[4, 19, 57, 91], 'score'] = np.nan
df.loc[[8, 42, 76], 'work_year'] = np.nan
df.loc[[10, 31], 'salary'] = np.nan

print(df.head())
print(df.info())
# 1-1

# score 결측치는 평균으로 대체하고, work_year 결측치가 있는 행은 삭제하시오.
# 이후 salary의 3사분위수를 구하시오.
# 정수로 출력하시오.


df['score'] = df['score'].fillna(df['score'].mean())
df = df.dropna(subset  ='work_year')
print(f"{int(df['salary'].quantile(0.75))}")
# 1-2

# join_date에서 연도와 월을 추출하시오.
# 2020년 이후 입사자 중 score가 상위 15명인 직원들의 평균 salary를 구하시오.
# 소수 둘째 자리까지 출력하시오.

df['join_date'] = pd.to_datetime(df['join_date'])
df['year'] = df['join_date'].dt.year
df['month'] = df['join_date'].dt.month
print(df[df['year'] >= 2020].sort_values('score', ascending = False).head(15)['salary'].mean())
# 1-3

# comment의 단어 수를 계산한 word_count 컬럼을 만드시오.
# dept별 평균 word_count를 구하고, 평균이 가장 낮은 부서명을 출력하시오.

df['word_count'] = df['comment'].str.split(" ").str.len()
print(df.groupby('dept')['word_count'].mean())
print(df.groupby('dept')['word_count'].mean().idxmin())
# 1-4

# score의 IQR을 구하고, score가 Q3 + 1.5 * IQR보다 큰 행의 개수를 출력하시오.
Q1 = df['score'].quantile(0.25)
Q3 = df['score'].quantile(0.75)
IQR = Q3 - Q1
cnt = (df['score'] > Q3 + 1.5  * IQR).sum() 
print(cnt)
# 제2유형

# 아래 코드 실행 후 문제를 푸시오.

import pandas as pd
import numpy as np

np.random.seed(802)

n_train = 650
n_test = 280

train = pd.DataFrame({
    'customer_id': range(10000, 10000 + n_train),
    'age': np.random.randint(18, 75, n_train),
    'income': np.random.normal(4600, 1200, n_train).round(1),
    'visit_count': np.random.poisson(9, n_train),
    'use_minutes': np.random.normal(60, 20, n_train).round(1),
    'grade': np.random.choice(['basic', 'silver', 'gold'], n_train),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu'], n_train)
})

test = pd.DataFrame({
    'customer_id': range(20000, 20000 + n_test),
    'age': np.random.randint(18, 75, n_test),
    'income': np.random.normal(4600, 1200, n_test).round(1),
    'visit_count': np.random.poisson(9, n_test),
    'use_minutes': np.random.normal(60, 20, n_test).round(1),
    'grade': np.random.choice(['basic', 'silver', 'gold', 'vip'], n_test),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Jeju'], n_test)
})

score = (
    -0.03 * train['age']
    + 0.00045 * train['income']
    + 0.24 * train['visit_count']
    + 0.02 * train['use_minutes']
    + train['grade'].map({'basic': -0.4, 'silver': 0.2, 'gold': 0.8})
    + train['region'].map({'Seoul': 0.3, 'Busan': 0.0, 'Daegu': -0.2})
    + np.random.normal(0, 1.15, n_train)
)

prob = 1 / (1 + np.exp(-score))
train['target'] = (prob > np.quantile(prob, 0.55)).astype(int)

train.loc[[7, 88, 210], 'income'] = np.nan
train.loc[[12, 133], 'grade'] = np.nan
test.loc[[5, 90], 'income'] = np.nan
test.loc[[21], 'grade'] = np.nan

print(train.head())
print(test.head())
print(train['target'].value_counts())

# 학습용 데이터를 이용해 target을 예측하고 평가용 데이터에 대한 결과를 제출하시오.

# 평가 지표: AUC
# 제출 파일명: result.csv
# 제출 컬럼: customer_id, pred
# pred: target이 1일 확률
# 제출 행 수는 평가용 데이터 행 수와 같아야 함

print(train.shape, train['customer_id'].nunique())
X = train.drop(['customer_id', 'target'], axis = 1)
y = train['target']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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
    X, y, test_size = 0.3, random_state = 42, stratify = y
)

pipe.fit(train_X, train_y)
prob = pipe.predict_proba(valid_X)[:, 1]
from sklearn.metrics import roc_auc_score
print("AUC: ", roc_auc_score(valid_y, prob))
pipe.fit(X,y)
customer_id = test['customer_id']
test_X = test.drop(['customer_id', 'target'], axis = 1, errors='ignore')
prob = pipe.predict_proba(test_X)[:, 1]
prob = pd.DataFrame({'customer_id': customer_id, 'pred': prob})
prob.to_csv("result.csv", index=  False)

import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(803)

df = pd.DataFrame({
    'group': np.repeat([0, 1], [48, 52]),
    'value': np.concatenate([
        np.random.lognormal(mean=2.0, sigma=0.30, size=48),
        np.random.lognormal(mean=2.25, sigma=0.45, size=52)
    ])
})

print(df.head())
print(df['group'].value_counts())

# value를 로그 변환한 뒤 group에 따라 두 집단으로 나누시오.

# 두 집단의 분산비 통계량을 구하시오. 큰 분산을 작은 분산으로 나누시오. 소수 셋째 자리까지 출력하시오.
# 두 집단의 합동분산 추정량을 구하시오. 소수 셋째 자리까지 출력하시오.
# 두 집단의 평균 차이에 대한 p-value를 구하시오. 소수 셋째 자리까지 출력하시오.

df['value'] = np.log(df['value'])
first = df[df['group'] == 0]
second = df[df['group'] == 1]
first_var = first['value'].var()
second_var = second['value'].var()
stat = max(first_var, second_var)/min(first_var, second_var)
print(f"{stat:.3f}")

len1 = len(first['value'])
len2 = len(second['value'])
var = ((len1 - 1) * first_var + (len2-1) * second_var)/ (len1 + len2 - 2)
print(f"{var:.3f}")

stat, pvalue = stats.ttest_ind(first['value'], second['value'], equal_var = True)
print(f"{pvalue:.3f}")



import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(804)

df = pd.DataFrame({
    'x': np.random.normal(30, 6, 90),
    'y': np.random.normal(100, 15, 90)
})

df['y'] = 40 + df['x'] * 2.1 + np.random.normal(0, 8, 90)

print(df.head())
# x와 y의 상관계수를 구하시오. 소수 셋째 자리까지 출력하시오.
# 위 관계의 p-value를 구하시오. 소수 셋째 자리까지 출력하시오.

stat, pvalue = stats.pearsonr(df['x'], df['y'])
print(f"{stat:.3f}")
print(f"{pvalue:.3f}")




import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(805)

n = 180

df = pd.DataFrame({
    'x1': np.random.normal(10, 2, n),
    'x2': np.random.normal(50, 8, n),
    'x3': np.random.normal(100, 20, n),
    'x4': np.random.normal(5, 1.5, n)
})

df['y'] = (
    30
    + 2.5 * df['x1']
    - 1.4 * df['x2']
    + 0.75 * df['x3']
    + 4.0 * df['x4']
    + np.random.normal(0, 8, n)
)

print(df.head())

# y를 종속변수로 하고 나머지를 독립변수로 하는 모형을 적합하시오
# 유의확률이 0.05 미만인 변수들의 회귀계수 합을 구하시오. 단, 절편은 제외한다. 소수 셋째 자리까지 출력하시오.

# 유의확률이 0.05 미만인 변수만 사용하여 다시 적합하시오
# . 수정 결정계수를 구하시오. 소수 셋째 자리까지 출력하시오.

# x1=12, x2=45, x3=110, x4=6인 데이터의 예측값을 구하시오
# . 단, 2번에서 선택된 변수만 사용한다. 소수 셋째 자리까지 출력하시오.

X = df.drop(columns = 'y')
y = df['y']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
pvalues = model.pvalues.drop('const')
print(f"{model.params.drop('const')[pvalues < 0.05].sum():.3f}")

sig_vars = pvalues[pvalues < 0.05].index
X = X[list(sig_vars)]
y = df['y']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(f"{model.rsquared:.3f}")

new_data = pd.DataFrame({'x1' : [12], 'x2' : [45], 'x3' : [110], 'x4': [6]})
new_data = new_data[list(sig_vars)]
new_data = sm.add_constant(new_data, has_constant = 'add')

print(f"{model.predict(new_data).iloc[0]:.3f}")