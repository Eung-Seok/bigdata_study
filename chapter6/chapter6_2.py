import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

admission_data = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/admission.csv')
print(admission_data.shape)
print(admission_data.head())

p_hat = admission_data['admit'].mean()
print(np.round(p_hat / (1-p_hat), 3))

unique_ranks = sorted(admission_data['rank'].unique())
print(unique_ranks)

grouped_data = admission_data.groupby('rank').agg(p_admit = ('admit', 'mean'))
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])
print(grouped_data)
print(np.round(1.178 / (1.178 + 1 ), 3))

# p = np.arange(0, 1.01, 0.01)
# log_odds = np.log(p / (1 - p))

# plt.plot(p, log_odds)
# plt.xlabel('p')
# plt.ylabel('log_odds')
# plt.title('Plot of log odds')
# plt.show()

odds_data = admission_data.groupby('rank').agg(p_admit = ('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
print(odds_data)

model = smf.ols('log_odds ~ rank', data = odds_data).fit()
print(model.summary())

# plt.scatter(odds_data['rank'], odds_data['log_odds'], label = 'Data Points')

# x = odds_data['rank']
# y = odds_data['log_odds']
# coefficients = np.polyfit(x,y,1)
# poly_eq = np.poly1d(coefficients)
# plt.plot(x, poly_eq(x), color = 'red', label = 'Regression Line')
# plt.xlabel('Rank')
# plt.ylabel('Log Odds')
# plt.title('Scatter Plot with Regression Line')
# plt.legend()
# plt.show()

selected_data = odds_data[['rank', 'p_admit', 'odds']]
selected_data['odds_frac'] = selected_data['odds'] / selected_data['odds'].shift(1, fill_value = selected_data['odds'].iloc[0])
print(selected_data)

rank_vec = np.array([1,2,3,4])
result = np.exp(0.6327 - 0.5675 * rank_vec) / (1 + np.exp(0.6327 - 0.5675 * rank_vec))
print(result)

selected_data = odds_data[['rank', 'p_admit', 'odds']]
selected_data['odds_frac'] = selected_data['odds']/ selected_data['odds'].shift(1, fill_value = selected_data['odds'].iloc[0])
print(selected_data)


rank_vec = np.array([1,2,3,4])
result = np.exp(0.6327 - 0.5675 * rank_vec) / (1 + np.exp(0.6327 - 0.5675 * rank_vec))
print(result)

admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')

model = smf.logit('admit ~ gre + gpa + rank + gender', data = admission_data).fit()
print(model.summary())

model = smf.glm('admit ~ gre + gpa + rank + gender', data = admission_data,
                family = sm.families.Binomial()).fit()
admission_data = pd.get_dummies(admission_data, columns = ['rank', 'gender'], drop_first = True)
admission_data[['rank_2', 'rank_3', 'rank_4', 'gender_M']] = admission_data[['rank_2', 'rank_3', 'rank_4', 'gender_M']].astype(int)

X = admission_data[['gre', 'gpa', 'rank_2', 'rank_3', 'rank_4', 'gender_M']]
y = admission_data['admit']

X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
print(model.summary())
model = sm.GLM(y, X, family = sm.families.Binomial()).fit()
print(model.summary())

admission_data = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/admission.csv')

admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')
model = smf.logit('admit ~ gre + gpa + rank + gender', data= admission_data).fit()
print(model.summary())

result1 = 0.002256 / 0.001094
result2 = 2 *(1 - stats.norm.cdf(result1))
print(result1)
print(result2)

odds_ratios = pd.DataFrame(
    {
        'OR' : model.params,
        'Lower CI' : model.conf_int()[0],
        'Upper CI' : model.conf_int()[1]
    }
)
odds_ratios = np.exp(odds_ratios)
print(odds_ratios)

a = round(model.params[5] - stats.norm.ppf(0.975) * 0.001094, 3)
b = round(model.params[5] + stats.norm.ppf(0.975) * 0.001094, 3)

glue_str = f'({a}, {b})'
print(glue_str)

model = smf.logit('admit ~ gre + gpa + rank + gender', data = admission_data).fit()
print(model.summary())

a = round(np.exp(a), 3)
b = round(np.exp(b), 3)

glue_str = f'({a}, {b})'
print(glue_str)

model = smf.logit('admit ~ gre + gpa + rank + gender', data = admission_data).fit()
print(model.summary())
print(model.llf)
print(model.llnull)

test_statistic = np.round(-2 * (model.llnull - model.llf), 3)
print('Test Statistic: ', test_statistic)

df = model.df_model - 0
p_value = chi2.sf(test_statistic, df)
print('p-value: ', np.round(p_value, 10))

admission_data = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/admission.csv')
admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')

model = smf.glm(formula = 'admit ~ gre + gpa + rank + gender', data = admission_data, family = sm.families.Binomial()).fit()

test_statistic2 = np.round(model.null_deviance - model.deviance, 3)
print(test_statistic2)

llf = model.llf
llnull = model.llnull
deviance = np.round(model.deviance, 3)
null_deviance = np.round(model.null_deviance,3)

deviance_calculated = np.round(-2 * llf, 3)
null_deviance_calculated = np.round(-2 * llnull, 3)

result = {
    'deviance == -2 * llf': deviance == deviance_calculated,
    'null_deviance == -2 * llnull' : null_deviance == null_deviance_calculated 
}

print(result)

new_data = pd.DataFrame({
    'gre' : [400,700,750,500],
    'gpa' : [3.5,3.8,3.9,3.2],
    'rank' : [2,1,4,3],
    'gender' : ['M','F','F','M']
})
y_true = pd.Series([0,1,0,0])

new_data['admit_prob'] = model.predict(new_data)
auc_score = roc_auc_score(y_true, new_data['admit_prob'])

print(new_data[['gre', 'gpa', 'rank', 'gender', 'admit_prob']])
print('AUC score: ', auc_score)