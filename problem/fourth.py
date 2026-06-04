# 문제 1
import pandas as pd
import numpy as np

np.random.seed(101)

df = pd.DataFrame({
    'id': range(1, 61),
    'store': np.random.choice(['A', 'B', 'C', 'D'], 60),
    'category': np.random.choice(['food', 'clothes', 'digital'], 60),
    'age': np.random.randint(18, 65, 60),
    'quantity': np.random.randint(1, 10, 60),
    'price': np.random.normal(30000, 8000, 60).round(0),
    'rating': np.random.normal(4.0, 0.6, 60).round(1)
})

df.loc[[3, 15, 27], 'price'] = np.nan
df.loc[[7, 22], 'rating'] = np.nan
df.loc[[10, 35], 'category'] = np.nan

print(df.head())
print(df.info())
# 1-1 price 결측치를 중앙값으로 대체한 뒤, price 평균을 소수 둘째 자리까지 출력하시오.

# 1-2 category 결측치를 최빈값으로 대체한 뒤, category별 평균 rating을 구하고 평균 rating이 가장 높은 category를 출력하시오.

# 1-3 rating 결측치를 중앙값으로 대체한 뒤, age >= 40이고 rating >= 4.2인 행의 개수를 출력하시오.

# 1-4 total = quantity * price 컬럼을 만들고, total이 가장 큰 상위 7개의 평균을 출력하시오.

# 1-5 store별 총 quantity를 구하고, 총 quantity가 가장 큰 store를 출력하시오.

# 문제 2
import pandas as pd
import numpy as np

np.random.seed(202)

n_train = 400
n_test = 150

train = pd.DataFrame({
    'product_id': range(1000, 1000 + n_train),
    'brand': np.random.choice(['A', 'B', 'C', 'D'], n_train),
    'type': np.random.choice(['basic', 'plus', 'premium'], n_train),
    'weight': np.random.normal(10, 2, n_train).round(2),
    'size': np.random.normal(50, 10, n_train).round(1),
    'defect_count': np.random.poisson(1.0, n_train),
    'production_year': np.random.randint(2015, 2025, n_train)
})

test = pd.DataFrame({
    'product_id': range(5000, 5000 + n_test),
    'brand': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_test),
    'type': np.random.choice(['basic', 'plus', 'premium', 'special'], n_test),
    'weight': np.random.normal(10, 2, n_test).round(2),
    'size': np.random.normal(50, 10, n_test).round(1),
    'defect_count': np.random.poisson(1.0, n_test),
    'production_year': np.random.randint(2015, 2025, n_test)
})

brand_effect = {'A': 200, 'B': 100, 'C': 0, 'D': -80}
type_effect = {'basic': 0, 'plus': 120, 'premium': 300}

train['target'] = (
    1000
    + train['weight'] * 40
    + train['size'] * 8
    - train['defect_count'] * 70
    + (train['production_year'] - 2015) * 30
    + train['brand'].map(brand_effect)
    + train['type'].map(type_effect)
    + np.random.normal(0, 120, n_train)
).round(1)

train.loc[[5, 33, 120], 'weight'] = np.nan
train.loc[[8, 77], 'brand'] = np.nan
test.loc[[2, 44], 'weight'] = np.nan
test.loc[[9], 'brand'] = np.nan

print(train.head())
print(test.head())
print(train.info())
# 2-1 target을 예측하는 모델을 만드시오.

# 조건:
# product_id 제거 여부를 판단하시오.
# 숫자형 결측치는 평균으로 대체
# 범주형 결측치는 최빈값으로 대체
# 범주형 변수는 인코딩 처리
# test 데이터에 train에 없던 범주가 있어도 에러가 나지 않게 처리
# random_state=42
# test_size=0.3
# 검증 데이터 RMSE를 출력하시오.

# 2-2 전체 train 데이터로 다시 학습한 뒤, test 데이터 예측값을 result_reg3.csv로 저장하시오.
# 컬럼명은 pred로 하시오.

# 2-3 product_id를 포함한 경우와 제외한 경우의 RMSE를 비교하고, 어떤 쪽을 선택할지 한 문장으로 설명하시오.

# 문제 3
import pandas as pd
import numpy as np

np.random.seed(303)

n_train = 450
n_test = 160

train = pd.DataFrame({
    'user_no': range(10000, 10000 + n_train),
    'age': np.random.randint(18, 70, n_train),
    'income': np.random.normal(4000, 1000, n_train).round(1),
    'visit': np.random.poisson(6, n_train),
    'use_time': np.random.normal(40, 15, n_train).round(1),
    'grade': np.random.choice(['normal', 'silver', 'gold'], n_train),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu'], n_train)
})

test = pd.DataFrame({
    'user_no': range(20000, 20000 + n_test),
    'age': np.random.randint(18, 70, n_test),
    'income': np.random.normal(4000, 1000, n_test).round(1),
    'visit': np.random.poisson(6, n_test),
    'use_time': np.random.normal(40, 15, n_test).round(1),
    'grade': np.random.choice(['normal', 'silver', 'gold', 'vip'], n_test),
    'region': np.random.choice(['Seoul', 'Busan', 'Daegu', 'Jeju'], n_test)
})

score = (
    -0.025 * train['age']
    + 0.0005 * train['income']
    + 0.25 * train['visit']
    + 0.03 * train['use_time']
    + train['grade'].map({'normal': -0.4, 'silver': 0.2, 'gold': 0.7})
    + train['region'].map({'Seoul': 0.3, 'Busan': 0.0, 'Daegu': -0.2})
    + np.random.normal(0, 1, n_train)
)

prob = 1 / (1 + np.exp(-score))
train['target'] = (prob > np.quantile(prob, 0.5)).astype(int)

train.loc[[6, 55, 99], 'income'] = np.nan
train.loc[[12, 80], 'grade'] = np.nan
test.loc[[4, 60], 'income'] = np.nan
test.loc[[15], 'grade'] = np.nan

print(train.head())
print(test.head())
print(train['target'].value_counts())

# 3-1 target을 예측하는 모델을 만드시오.

# 조건:
# user_no 제거 여부를 판단하시오.
# 숫자형 결측치는 평균으로 대체
# 범주형 결측치는 최빈값으로 대체
# 범주형 변수는 인코딩 처리
# test 데이터에 train에 없던 범주가 있어도 에러가 나지 않게 처리
# random_state=42
# test_size=0.3
# 검증 데이터의 accuracy, f1_macro, AUC를 출력하시오.

# 3-2 평가 기준이 f1_macro일 때, test 예측값을 result_cls3.csv로 저장하시오.
# 컬럼명은 pred.

# 3-3 평가 기준이 AUC일 때, test 예측값을 result_auc3.csv로 저장하시오.
# 컬럼명은 pred.

# 문제 4
import numpy as np
from scipy import stats

data = np.array([52, 55, 49, 58, 54, 56, 51, 57, 53, 55, 59, 54])

# 이 데이터의 평균이 53보다 크다고 할 수 있는지 유의수준 0.05에서 판단하시오.

# 출력:
# 검정통계량
# p-value
# 귀무가설
# 대립가설
# 결론

# 문제 5

import numpy as np
from scipy import stats

A = np.array([72, 75, 78, 74, 77, 79, 73, 76])
B = np.array([70, 68, 69, 71, 67, 72, 70, 69])

# 두 그룹의 평균에 차이가 있는지 유의수준 0.05에서 판단하시오.
# 출력:
# 검정통계량
# p-value
# 귀무가설
# 대립가설
# 결론

# 문제 6
import numpy as np
from scipy import stats

before = np.array([80, 82, 78, 85, 83, 81, 79, 84])
after = np.array([78, 80, 75, 82, 80, 79, 76, 81])

# 동일한 대상의 전후 측정값이다. 이후 값이 이전 값보다 감소했다고 할 수 있는지 유의수준 0.05에서 판단하시오.
# 출력:
# 검정통계량
# p-value
# 귀무가설
# 대립가설
# 결론

# 문제 7
import numpy as np
from scipy import stats

table = np.array([
    [35, 25],
    [20, 40],
    [30, 30]
])

# 행은 지역 A, B, C이고 열은 선택함, 선택하지 않음이다.
# 지역과 선택 여부가 서로 관련 있다고 볼 수 있는지 유의수준 0.05에서 판단하시오.
# 출력:
# 검정통계량
# p-value
# 자유도
# 기대빈도
# 귀무가설
# 대립가설
# 결론

# 문제 8
import numpy as np
from scipy import stats

observed = np.array([16, 21, 19, 24, 20])
expected = np.repeat(20, 5)

# 관측된 빈도가 기대되는 빈도와 차이가 있다고 볼 수 있는지 유의수준 0.05에서 판단하시오.
# 출력:
# 검정통계량
# p-value
# 귀무가설
# 대립가설
# 결론

# 문제 9
import numpy as np
from scipy import stats

g1 = np.array([61, 63, 62, 64, 65])
g2 = np.array([70, 72, 71, 73, 74])
g3 = np.array([66, 67, 65, 68, 69])
g4 = np.array([80, 79, 81, 82, 83])

# 네 그룹의 평균이 모두 같다고 볼 수 있는지 유의수준 0.05에서 판단하시오.
# 출력:
# 검정통계량
# p-value
# 귀무가설
# 대립가설
# 결론

# 문제 10
import numpy as np
from scipy import stats

sample = np.array([10.2, 9.8, 10.5, 10.1, 9.9, 10.3, 10.4, 9.7, 10.0, 10.2])

# 위 데이터가 정규분포를 따른다고 볼 수 있는지 유의수준 0.05에서 판단하시오.
# 출력:
# 검정통계량
# p-value
# 귀무가설
# 대립가설
# 결론

# 문제 11
import numpy as np
from scipy import stats

x1 = np.array([45, 47, 46, 48, 49, 47])
x2 = np.array([50, 55, 52, 58, 57, 54])

# 두 그룹의 분산이 같다고 볼 수 있는지 유의수준 0.05에서 판단하시오.
# 출력:
# 검정통계량
# p-value
# 귀무가설
# 대립가설
# 결론

# 문제 12
import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(404)

n = 160

df = pd.DataFrame({
    'x1': np.random.normal(10, 2, n),
    'x2': np.random.normal(30, 5, n),
    'x3': np.random.normal(100, 20, n),
    'x4': np.random.normal(5, 1, n)
})

df['y'] = (
    20
    + 3.5 * df['x1']
    - 1.8 * df['x2']
    + 0.7 * df['x3']
    + 5.0 * df['x4']
    + np.random.normal(0, 8, n)
)

print(df.head())

# y를 종속변수로 하고 나머지를 독립변수로 하는 모형을 적합하시오.
# 출력:
# summary
# R-squared
# 수정 R-squared
# p-value가 가장 큰 변수명과 p-value
# 유의수준 0.05에서 유의한 변수 개수
# 회귀계수가 가장 큰 변수명과 값
# x1=12, x2=28, x3=110, x4=6일 때 예측값

# 문제 13
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

np.random.seed(505)

n = 280

df = pd.DataFrame({
    'a': np.random.normal(5, 1.5, n),
    'b': np.random.normal(70, 10, n),
    'c': np.random.normal(40, 8, n),
    'd': np.random.poisson(2, n)
})

z = (
    -3
    + 0.6 * df['a']
    + 0.025 * df['b']
    + 0.02 * df['c']
    - 0.4 * df['d']
    + np.random.normal(0, 1.5, n)
)

prob = 1 / (1 + np.exp(-z))
df['target'] = (prob > 0.55).astype(int)

print(df.head())
print(df['target'].value_counts())
# 13-1 target을 종속변수로 하고 a만 독립변수로 하는 모형을 적합한 뒤, a의 오즈비를 출력하시오.

# 13-2 target을 종속변수로 하고 모든 변수를 독립변수로 하는 모형을 적합한 뒤, residual deviance를 출력하시오.

# 13-3 모든 변수를 사용한 모형에서 p-value가 0.05 이상인 변수 개수를 출력하시오.
# 단, 상수항은 제외하시오.

# 13-4 p-value가 0.05 미만인 변수만 사용해서 다시 적합하고, 해당 변수들의 회귀계수 평균을 출력하시오.
# 단, 상수항은 평균 계산에서 제외하시오.

# 13-5 데이터를 학습/평가로 나누시오.

# 조건:
# test_size=80
# random_state=42
# 모든 변수를 사용하여 평가 데이터의 오분류율을 출력하시오.
# 예측확률이 0.5 이상이면 1로 판단하시오.

# 13-6 13-5의 평가 데이터에서 AUC를 출력하시오.