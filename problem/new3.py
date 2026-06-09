import pandas as pd
import numpy as np

np.random.seed(910)

df = pd.DataFrame({
    '학생ID': range(1001, 1131),
    '반': np.random.choice(['A', 'B', 'C', 'D'], 130),
    '성별': np.random.choice(['M', 'F'], 130),
    '시험일': pd.date_range('2021-01-01', periods=130, freq='15D'),
    '국어': np.random.normal(75, 12, 130).round(1),
    '수학': np.random.normal(70, 15, 130).round(1),
    '영어': np.random.normal(78, 10, 130).round(1),
    '출석일수': np.random.randint(60, 101, 130),
    '메모': np.random.choice([
        'good study habit',
        'late homework issue',
        'excellent class attitude',
        'normal participation',
        'needs more practice'
    ], 130)
})

df.loc[[5, 22, 47, 88], '수학'] = np.nan
df.loc[[13, 51, 90], '영어'] = np.nan
df.loc[[7, 44], '국어'] = np.nan

print(df.head())
print(df.info())
# 1-1

# 수학 결측치는 중앙값으로 대체하고, 영어 결측치가 있는 행은 삭제하시오.
# 이후 국어의 1사분위수를 구하시오.
# 정수로 출력하시오.

df['수학'] = df['수학'].fillna(df['수학'].median())
df = df.dropna(subset = ['영어'])
print(int(np.round(df['국어'].quantile(0.25))))
# 1-2

# 시험일에서 연도와 월을 추출하시오.
# 2022년 이후 데이터 중 출석일수가 상위 12명인 학생들의 평균 수학 점수를 구하시오.
# 소수 둘째 자리까지 출력하시오.

df['시험일'] = pd.to_datetime(df['시험일'])
df['연도'] = df['시험일'].dt.year
df['월'] = df['시험일'].dt.month
print(f"{df[df['연도'] >= 2022].sort_values('출석일수', ascending = False).head(12)['수학'].mean():.2f}")
# 1-3

# 메모 컬럼의 단어 수를 계산한 word_count 컬럼을 만드시오.
# 반별 평균 word_count를 구하고, 평균이 두 번째로 낮은 반을 출력하시오.

df['word_count'] = df['메모'].str.split().str.len()
print(df.groupby('반')['word_count'].mean())
print(df.groupby('반')['word_count'].mean().sort_values().index[1])

# 1-4
# 국어, 수학, 영어 평균으로 평균점수 컬럼을 만드시오.
# 평균점수의 IQR을 구하고, 평균점수가 Q1 - 1.5 * IQR보다 작은 행의 개수를 출력하시오.

df['평균점수'] = (df['국어'] + df['수학'] + df['영어'])/3
Q1 = df['평균점수'].quantile(0.25)
Q3 = df['평균점수'].quantile(0.75)
IQR = Q3 - Q1
cnt = (df['평균점수'] < Q1 - 1.5 * IQR).sum()
print(cnt)
# 작업형 제2유형

# 아래 데이터로 예측 결과를 생성하시오.

import pandas as pd
import numpy as np

np.random.seed(920)

n_train = 720
n_test = 310

train = pd.DataFrame({
    'cust_id': range(10000, 10000 + n_train),
    'age': np.random.randint(18, 76, n_train),
    'income': np.random.normal(4200, 1100, n_train).round(1),
    'visit': np.random.poisson(7, n_train),
    'use_time': np.random.normal(45, 18, n_train).round(1),
    'grade': np.random.choice(['basic', 'silver', 'gold'], n_train),
    'channel': np.random.choice(['app', 'web', 'store'], n_train)
})

test = pd.DataFrame({
    'cust_id': range(20000, 20000 + n_test),
    'age': np.random.randint(18, 76, n_test),
    'income': np.random.normal(4200, 1100, n_test).round(1),
    'visit': np.random.poisson(7, n_test),
    'use_time': np.random.normal(45, 18, n_test).round(1),
    'grade': np.random.choice(['basic', 'silver', 'gold', 'vip'], n_test),
    'channel': np.random.choice(['app', 'web', 'store', 'phone'], n_test)
})

score = (
    -0.025 * train['age']
    + 0.0005 * train['income']
    + 0.25 * train['visit']
    + 0.025 * train['use_time']
    + train['grade'].map({'basic': -0.4, 'silver': 0.2, 'gold': 0.7})
    + train['channel'].map({'app': 0.3, 'web': 0.0, 'store': -0.2})
    + np.random.normal(0, 1.1, n_train)
)

prob = 1 / (1 + np.exp(-score))
train['target'] = (prob > np.quantile(prob, 0.55)).astype(int)

train.loc[[8, 77, 201], 'income'] = np.nan
train.loc[[15, 130], 'grade'] = np.nan
test.loc[[4, 91], 'income'] = np.nan
test.loc[[21], 'grade'] = np.nan

print(train.head())
print(test.head())
print(train['target'].value_counts())

# 학습용 데이터를 이용해 target을 예측하고 평가용 데이터에 대한 결과를 제출하시오.

# 평가 지표: AUC
# 제출 파일명: result.csv
# 제출 컬럼: cust_id, pred
# pred: target이 1일 확률
# 제출 행 수는 평가용 데이터 행 수와 같아야 함

print(train.shape, train['cust_id'].nunique())
X = train.drop(['target', 'cust_id'], axis = 1)
y = train['target']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
pipe.fit(train_X, train_y)
prob = pipe.predict_proba(valid_X)[:, 1]
print("AUC: ", roc_auc_score(valid_y, prob))

pipe.fit(X,y)
cust_id = test['cust_id']
test_X = test.drop(['cust_id', 'target'], errors = 'ignore', axis = 1)
prob = pipe.predict_proba(test_X)[:, 1]
prob = pd.DataFrame({'cust_id' : cust_id, 'pred': prob})
prob.to_csv('result.csv', index = False)

# 작업형 제3유형
# 문제 1

# 아래 데이터로 문제를 푸시오.

import pandas as pd
import numpy as np

np.random.seed(930)

df = pd.DataFrame({
    'group': np.repeat([1, 2], [50, 60]),
    'value': np.concatenate([
        np.random.lognormal(mean=2.0, sigma=0.32, size=50),
        np.random.lognormal(mean=2.25, sigma=0.46, size=60)
    ])
})

print(df.head())
print(df['group'].value_counts())

# value를 로그 변환한 뒤 group에 따라 두 집단으로 나누시오.

# 두 집단의 분산비 통계량을 구하시오. 큰 분산을 작은 분산으로 나누시오. 소수 셋째 자리까지 출력하시오.
# 두 집단의 합동분산 추정량을 구하시오. 소수 셋째 자리까지 출력하시오.
# 두 집단 평균 차이에 대한 p-value를 구하시오. 소수 셋째 자리까지 출력하시오.

df['value'] = np.log(df['value'])

first = df[df['group'] == 1]
second = df[df['group'] == 2]

f_var = first['value'].var()
s_var = second['value'].var()
ratio = max(f_var, s_var) / min(f_var, s_var)
print(f"{ratio:.3f}")

f_len = len(first['value'])
s_len = len(second['value'])

var = ((f_len - 1) * f_var + (s_len - 1) * s_var)/(f_len + s_len - 2) 
print(f"{var:.3f}")

from scipy import stats
stat, pvalue = stats.ttest_ind(first['value'], second['value'], equal_var = True, alternative = 'two-sided')
print(f"{pvalue:.3f}")

# 문제 2

# 아래 데이터로 문제를 푸시오.

import pandas as pd
import numpy as np

np.random.seed(940)

df = pd.DataFrame({
    'hours': np.random.normal(5, 1.4, 90),
    'score': np.random.normal(70, 10, 90)
})

df['score'] = 45 + df['hours'] * 6.0 + np.random.normal(0, 7, 90)
df['score'] = df['score'].round(1)

print(df.head())
# hours와 score의 상관계수를 구하시오. 소수 셋째 자리까지 출력하시오.
# 위 관계의 p-value를 구하시오. 소수 셋째 자리까지 출력하시오.

from scipy import stats
stat, pvalue = stats.pearsonr(df['hours'], df['score'])
print(f"{stat:.3f}")
print(f"{pvalue:.3f}")

# 문제 3

# 아래 데이터로 문제를 푸시오.

import pandas as pd
import numpy as np

np.random.seed(950)

n = 200

df = pd.DataFrame({
    'area': np.random.normal(90, 18, n),
    'height': np.random.normal(11, 2.5, n),
    'wall': np.random.choice([0, 1], n),
    'age': np.random.normal(12, 4, n)
})

df['price'] = (
    9000
    + df['area'] * 180
    + df['height'] * 420
    + df['wall'] * 1500
    - df['age'] * 230
    + np.random.normal(0, 2400, n)
).round(1)

print(df.head())

# price를 종속변수로 하고 나머지를 독립변수로 하는 모형을 적합하시오. 유의확률이 0.05 미만인 변수들의 회귀계수 합을 구하시오. 단, 절편은 제외한다. 소수 셋째 자리까지 출력하시오.
# 유의확률이 0.05 미만인 변수만 사용하여 다시 모형을 적합하시오. 수정 결정계수를 구하시오. 소수 셋째 자리까지 출력하시오.
# area=100, height=12, wall=1, age=10인 데이터의 예측값을 구하시오. 단, 2번에서 선택된 변수만 사용한다. 소수 셋째 자리까지 출력하시오.

import statsmodels.api as sm
X = df.drop(columns = 'price')
y = df['price']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
pvalues = model.pvalues.drop('const')
print(f"{model.params.drop('const')[pvalues < 0.05].sum():.3f}")
sig_var = pvalues[pvalues < 0.05].index
X = X[list(sig_var)]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(f"{model.rsquared_adj:.3f}")
new_data = pd.DataFrame({'area' : [100], 'height' : [12], 'wall' : [1], 'age' : [10]})
new_data = new_data[list(sig_var)]
new_data = sm.add_constant(new_data, has_constant = 'add')
print(f"{model.predict(new_data).iloc[0]:.3f}")
# 문제 4

# 아래 데이터로 문제를 푸시오.

import pandas as pd
import numpy as np

np.random.seed(960)

n = 230

df = pd.DataFrame({
    'age': np.random.randint(20, 60, n),
    'income': np.random.normal(3600, 850, n),
    'overtime': np.random.choice([0, 1, 2], n)
})

z = (
    -3.2
    + 0.04 * df['age']
    - 0.00025 * df['income']
    + 0.75 * df['overtime']
    + np.random.normal(0, 1, n)
)

prob = 1 / (1 + np.exp(-z))
df['target'] = (prob >= 0.5).astype(int)

print(df.head())
print(df['target'].value_counts())
# target을 종속변수로 하고 나머지를 독립변수로 하는 모형을 적합하시오. 유의확률이 0.05 미만인 변수의 개수를 구하시오. 단, 절편은 제외한다.
# overtime이 1 증가할 때 target=1의 오즈비를 구하시오. 소수 셋째 자리까지 출력하시오.
# age=35, income=3200, overtime=2인 데이터의 target=1 예측확률을 구하시오. 소수 셋째 자리까지 출력하시오
import statsmodels.api as sm

X = df.drop(columns = 'target')
y = df['target']
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
cnt = (model.pvalues.drop('const') < 0.05).sum()
print(cnt)
print(f"{np.exp(model.params['overtime']):.3f}")

new_data = pd.DataFrame({'age' : [35], 'income' : [3200], 'overtime' : [2]})
new_data = sm.add_constant(new_data, has_constant = 'add')
print(f"{model.predict(new_data).iloc[0]:.3f}")