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
correlation_matrix = df.corr()
target_corr = correlation_matrix['target'].drop('target')
max_corr_var = target_corr.abs().idxmax()
max_corr_value = target_corr.abs().max()
print(f'가장 큰 상관계수를 갖는 변수: {max_corr_var}, 상관계수: {max_corr_value}')
#2 다중 선형회귀 모형으로 target 변수를 예측할 때, 모델의 결정계수를 계산하시오.
X = df.drop(columns = 'target')
y = df['target']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

r_squared = model.rsquared
print(f'결정계수: {r_squared}')
#3 앞에서 사용된 모델의 계수 검정에서 p-value가 가장 큰 변수와 그 값을 구하시오.
p_values = model.pvalues.drop('const')
max_p_value_var = p_values.idxmax()
max_p_value = p_values.max()
print(f'가장 큰 p-value를 갖는 변수: {max_p_value_var}, p-value: {max_p_value}')


X, y = make_regression(n_samples = 100, n_features = 3, noise = 0.1, random_state = 42)
df = pd.DataFrame(X, columns= [f'var{i}' for i in range(3)])
df['target'] = y
print(df.head())
#4 유의확률(p-value)이 가장 작은 변수의 회귀계수를 구하시오.
X = df.drop(columns = 'target')
y = df['target']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

p_values = model.pvalues
smallest_p_var = p_values.idxmin()
smallest_p_coef = model.params[smallest_p_var]
print(f'p값이 제일 작은 변수의 회귀계수: {smallest_p_coef}')
#5 적합된 회구모델의 결정계수를 구하시오.
r_squared = model.rsquared
print(f'결정계수: {r_squared}')
#6 적합된 회귀모델을 사용하여 var0 변수가 0.5, var1은 1.2, 그리고 var2는 0.3일 때 예측값을 계산하시오.
new_data = pd.DataFrame({'const' : [1.0], 'var0' : [0.5], 'var1' : [1.2], 'var2' : [0.3]})
predicted_value = model.predict(new_data)
print(f'예측된 값: {predicted_value[0]}')