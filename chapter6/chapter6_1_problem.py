import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import statsmodels.api as sm
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(n_samples)
df = pd.DataFrame(X, columns = ['var1', 'var2', 'var3', 'var4', 'var5'])
df['target'] = y
print(df.head())
#1 target변수와 가장 큰 상관 관계를 갖는 변수의 상관계수를 구하시오.
#2 다중 선형회귀 모형으로 target 변수를 예측할 때, 모델의 결정계수를 계산하시오.
#3 앞에서 상ㅇ된 모델의 계수 검정에서 p-value가 가장 큰 변수와 그 값을 구하시오.

X, y = make_regression(n_samples = 100, n_features = 3, noise = 0.1, random_state = 42)
df = pd.DataFrame(X, columns= [f'var{i}' for i in range(3)])
df['target'] = y
print(df.head())
#4 유의확률(p-value)이 가장 작은 변수의 회귀계수를 구하시오.
#5 적합된 회구모델의 결정계수를 구하시오.
#6 적합된 회귀모델을 사용하여 var0 변수가 0.5, var1은 1.2, 그리고 var2는 0.3일 때 예측값을 계산하시오.