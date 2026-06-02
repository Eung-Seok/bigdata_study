import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error

x = np.array([10,20,30,40,50])
y = np.array([5,15,25,35,48])

corr_coeff, p_value = stats.pearsonr(x,y)

print(f'피어슨 상관계수 (r): {corr_coeff:.4f}')
print(f'p-value: {p_value:.4f}')

x_mean = np.mean(x)
y_mean = np.mean(y)

numerator = np.sum((x - x_mean) * (y - y_mean))
denomirator = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))

r = numerator / denomirator
print(f'피어슨 상관계수 (수식 기반): {r:.4f}')

Sxy = np.sum((x - x_mean) * (y - y_mean)) 
Sxx = np.sum((x - x_mean)**2)
beta_1 = Sxy / Sxx

beta_0 = y_mean - beta_1 * x_mean

print(f'기울기 (b1): {beta_1:.4f}')
print(f'절편 (b0): {beta_0:.4f}')

print(f'회귀직선 방정식: y = {beta_0:.4f} + {beta_1:.4f} * x')

y_pred = beta_0 + beta_1 * x

# plt.figure(figsize=(6,4))
# plt.scatter(x,y, label = 'real', color='blue')
# plt.plot(x,y_pred,
#          color='red',
#          label='fitted line')
# plt.title('fitted line visualization')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()

s_x, s_y = np.std(x, ddof = 1), np.std(y, ddof = 1)
r, _ = pearsonr(x, y)

beta_1 = r * (s_y/s_x)
beta_0 = y_mean - beta_1 * x_mean
print(f'상관계수 r: {r: .4f}')
print(f'기울기 b1 (r * sy/sx): {beta_1:.4f}')

df_iris = load_iris()

iris = pd.DataFrame(data = df_iris.data, columns = df_iris.feature_names)
iris.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

iris['species'] = df_iris.target
iris['species'] = iris['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

model = smf.ols('Petal_Length ~ Petal_Width + Sepal_Length', data = iris).fit()
print(model.summary())

model = smf.glm('Petal_Length ~ Petal_Width + Sepal_Length', family = sm.families.Gaussian(), data = iris).fit()
print(model.summary())

model = smf.ols('Petal_Length ~ Petal_Width + Sepal_Length + C(species)', data = iris).fit()
print(model.summary())

X = iris[['Petal_Width', 'Sepal_Length']]
y = iris['Petal_Length']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

X = iris[['Petal_Width', 'Sepal_Length', 'species']]

X = pd.get_dummies(X, columns = ['species'], drop_first= True)
X = X.astype(float)
y = iris['Petal_Length']
X = sm.add_constant(X)
model2 = sm.OLS(y, X).fit()

coefficients = model.params[1:]
print('회귀계수가 가장 큰 변수: ', coefficients.idxmax())

t_values = model.tvalues
print('t-values: \n', t_values)

p_values = model.pvalues
print('p-values: \n', p_values)
print('|tvalue|가 가장 큰 변수: ', np.abs(t_values).idxmax())

conf_intervals = model.conf_int()
print('Confidence intervals: \n', conf_intervals)

conf_intervals_90 = model.conf_int(alpha = 0.10)
print('90% Confidence intervals: \n', conf_intervals_90)

data = {
    'color' : ['red', 'blue', 'green', 'red', 'green', 'red', 'green', 'blue', 'green', 'red'],
    'size' : [1,2,3,1,3,5,9,2,9,10],
    'price' : [10,20,30,10,30,55,29,10,25,12]
}
df = pd.DataFrame(data)
df_dummies = pd.get_dummies(df, columns = ['color'], drop_first = True)
print(df_dummies)

X = df_dummies[['size', 'color_green', 'color_red']]
y = df_dummies['price']

X = X.astype(float)
y = y.astype(float)

X = sm.add_constant(X)

model2 = sm.OLS(y, X).fit()
print(model2.summary())

formula = 'price ~ size + C(color)'
model2 = smf.ols(formula, data = df).fit()
print(model2.summary())

print('R-squared: ', np.round(model.rsquared, 2))
print('Adj. R-squared: ', np.round(model.rsquared_adj, 2))
print('F-statistic: ', np.round(model.fvalue, 4))
print('Prob (F-statistic): ', np.round(model.f_pvalue, 4))
print('AIC', np.round(model.aic, 2))
print('BIC', np.round(model.bic, 2))

model1 = ols('Petal_Length ~ Petal_Width', data = iris).fit()
model2 = ols('Petal_Length ~ Petal_Width + Sepal_Length + Sepal_Width', data = iris).fit()

table = sm.stats.anova_lm(model1, model2)
print(table)

dw_stat = model.summary().tables[2].data[0][3]
print(f'Durbin-Watson statistic: {dw_stat}')

dw_stat = durbin_watson(model.resid)
print(dw_stat)

residuals = model.resid
sw_stat, sw_p_value = shapiro(residuals)

print(f'Shapiro-Wilk Test Statistic: {sw_stat}')
print(f'p-value: {sw_p_value}')

bptest = het_breuschpagan(model.resid, model.model.exog)

print('BP-test statistics: ', bptest[0])
print('p-value: ', bptest[1])

X = iris[['Petal_Width', 'Sepal_Width']]

vif_data = pd.DataFrame()
vif_data['Variable'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

model = smf.ols('Petal_Length ~ Petal_Width + Sepal_Length + C(species)', data = iris).fit()

new_data = pd.DataFrame({
    'Petal_Width' : [0.2,1.5,1.3,2.1,1.8],
    'Sepal_Length' : [4.9,5.5,6.4,6.7,7.2],
    'species' : ['setosa', 'versicolor', 'virginica', 'versicolor', 'virginica']
})

y_pred = model.predict(new_data)
y_true = np.array([1.4,4.7,5.1,5.8,6.3])

mse_score = mean_squared_error(y_true, y_pred)
print('예측값:\n', y_pred)
print(f'MSE: {mse_score: .4f}')