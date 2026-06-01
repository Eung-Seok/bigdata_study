import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import scipy.stats as sp
from scipy.stats import bartlett
from statsmodels.stats.multicomp import pairwise_tukeyhsd

scents = ['Lavender', 'Rosemary', 'Peppermint']
minutes_lavender = [10,12,11,9,8,12,11,10,10,11]
minutes_rosemary = [14,15,13,16,14,15,14,13,14,16]
minutes_peppermint = [18,17,18,16,17,19,18,17,18,19]

anova_data = pd.DataFrame({
    'Scent' : np.repeat(scents,10),
    'Minutes' : minutes_lavender + minutes_rosemary + minutes_peppermint
})
print(anova_data.head())

print(anova_data.groupby(['Scent']).describe())

lavender = anova_data[anova_data['Scent'] == 'Lavender']['Minutes']
rosemary = anova_data[anova_data['Scent'] == 'Rosemary']['Minutes']
peppermint = anova_data[anova_data['Scent'] == 'Peppermint']['Minutes']

f_statistic, p_value = f_oneway(lavender, rosemary, peppermint)
print(f'F-statistic: {f_statistic}, p-value: {p_value}')

model = ols('Minutes ~ C(Scent)', data = anova_data).fit()

anova_results = sm.stats.anova_lm(model, typ = 2)
print(anova_results)

# plt.scatter(model.fittedvalues, model.resid)
# plt.show()

# sp.probplot(model.resid, dist = 'norm', plot = plt)
# plt.show()

W, p = sp.shapiro(model.resid)
print(f'검정통계랑: {W: .3f}, 유의확률: {p: .3f}')

groups = ['Lavender', 'Rosemary', 'Peppermint']
grouped_residuals = [model.resid[anova_data['Scent'] == group] for group in groups]

test_statistic, p_value = bartlett(*grouped_residuals)
print(f'검정통계량: {test_statistic}, p-value: {p_value}')

tukey = pairwise_tukeyhsd(endog = anova_data['Minutes'],
                           groups = anova_data['Scent'],
                           alpha = 0.05)

print(tukey)