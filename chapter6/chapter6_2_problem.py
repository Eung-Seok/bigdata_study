import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes


np.random.seed(42)
n_samples = 210
X = np.random.randn(n_samples, 4)
y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
df = pd.DataFrame(X, columns = ['weight', 'height', 'age', 'income'])
df['gender'] = y
print(df.head())

#  해당하는 오즈비(weight)를 계산하시오
#2 성별 변수를 주어진 4개 변수를 사용하여 로지스틱 회귀모델을 적합했을 때, residual deviance를 계산하시오.
#3 1번 문제의 모델(몸무게를 독립변수로 사용) 데이터를 학습 데이터와 평가데이터(90개로 설정)
#  로 분류한 후, 오분류율을 계산하시오.(소수점 넷째 자리에서 반올림)
# 분할 시 당므의 코드를 활용
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 90, random_state = 42)

diabetes = load_diabetes(as_frame = True)
df = diabetes.frame
print(df.head())

#4 target 변수를 중앙값을 기준으로 낮으면 0, 높으면 1로 이진화한 후, 로지스틱 회귀모델을 적합시키고
#  통계적으로 유의하지 않은 변수의 개수를 구하시오(조건: 유의수준은 0.05로 설정, 상수항 계수가 유의할
#  경우 변수 개수에 포함, s1~s6 변수 제거)
#5 4번 문제에서 유의한 변수들만 사용하여 다시 로지스틱 회귀 적합하고, 유의한 변수들의 회귀계수 평균을 구하시오.
#6 4번 문제에서 나이가 1 단위로 증가할 때 오즈비를 계산하시오.