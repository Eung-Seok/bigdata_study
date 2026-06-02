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

#1 성별 변수(gender)를 사용하여 몸무게 변수(weight)에 대한 로지스틱 회귀모델을 적합하고
#  해당하는 오즈비(weight)를 계산하시오
import statsmodels.api as sm

X_weight = df[['weight']]
X_weight = sm.add_constant(X_weight)
y = df['gender']
logit_model_weight = sm.Logit(y, X_weight).fit()
print(logit_model_weight.summary())

odds_ratio_weight = np.exp(logit_model_weight.params['weight'])
print(f'weight의 오즈비: {odds_ratio_weight}')
#2 성별 변수를 주어진 4개 변수를 사용하여 로지스틱 회귀모델을 적합했을 때, residual deviance를 계산하시오.
X_all = df[['weight', 'height', 'age', 'income']]
X_all = sm.add_constant(X_all)
logit_model_all = sm.Logit(y, X_all).fit()
residual_deviance = -2 * logit_model_all.llf
print(f'Residual deviance: {residual_deviance.round(3)}')
#3 1번 문제의 모델(몸무게를 독립변수로 사용) 데이터를 학습 데이터와 평가데이터(90개로 설정)
#  로 분류한 후, 오분류율을 계산하시오.(소수점 넷째 자리에서 반올림)
# 분할 시 당므의 코드를 활용
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 90, random_state = 42)

X_train = sm.add_constant(df_train[['weight']])
y_train = df_train['gender']
X_test = sm.add_constant(df_test[['weight']])
y_test = df_test['gender']

print(X_train.shape)
print(X_test.shape)

logit_model_train = sm.Logit(y_train, X_train).fit()

from sklearn.metrics import accuracy_score
y_pred = logit_model_train.predict(X_test) > 0.5
error_rate = 1 - accuracy_score(y_test,y_pred)
print(f'오분류율: {error_rate:.4}')
#------------------------------------------------------------------------------

diabetes = load_diabetes(as_frame = True)
df = diabetes.frame
print(df.head())

#4 target 변수를 중앙값을 기준으로 낮으면 0, 높으면 1로 이진화한 후, 로지스틱 회귀모델을 적합시키고
#  통계적으로 유의하지 않은 변수의 개수를 구하시오(조건: 유의수준은 0.05로 설정, 상수항 계수가 유의할
#  경우 변수 개수에 포함, s1~s6 변수 제거)
import statsmodels.api as sm
X = df.iloc[:, 0:4]
X = sm.add_constant(X)

y = (df['target'] > df['target'].median()).astype(int)

logit_model = sm.Logit(y,X).fit()
print(logit_model.summary())

p_values = logit_model.pvalues
non_significant_vars = p_values[p_values >= 0.05]
num_non_significant_vars = len(non_significant_vars)

print(f'유의미하지 않은 변수의 수: {num_non_significant_vars}')
#5 4번 문제에서 유의한 변수들만 사용하여 다시 로지스틱 회귀 적합하고, 유의한 변수들의 회귀계수 평균을 구하시오.
significant_vars = p_values[p_values < 0.05]
significant_vars_names = significant_vars.index.drop('const', errors = 'ignore')

X_significant = X[significant_vars_names]
X_significant = sm.add_constant(X_significant)

logit_model_significant = sm.Logit(y, X_significant).fit()
significant_coef_mean = logit_model_significant.params.mean()
print(f'유의한 변수들만 사용 시 회귀계수들의 평균: {significant_coef_mean}')
#6 4번 문제에서 나이가 1 단위로 증가할 때 오즈비를 계산하시오.
coef_age = logit_model.params['age']
delta_x = 1
odds_ratio = np.exp(coef_age * delta_x)
print(f'age변수가 1단위 증가할 때 오즈비: {odds_ratio}')