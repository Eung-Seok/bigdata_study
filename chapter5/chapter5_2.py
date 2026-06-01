import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
from scipy.stats import norm
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import scipy.stats as stats
from scipy.stats import f

sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45]
t_statistic, p_value = ttest_1samp(sample, popmean = 10, alternative = 'two-sided')
print('t-statistic: ', t_statistic)
print('p-value: ', p_value)

gender = ['Female'] * 7 + ['Male'] * 5

my_tab2 = pd.DataFrame({'score' : sample, 'gender': gender})
print(my_tab2)

male = my_tab2[my_tab2['gender'] == 'Male']
female = my_tab2[my_tab2['gender'] == 'Female']

t_statistic, p_value = ttest_ind(male['score'], female['score'],
                                 equal_var = True, alternative = 'greater')
print('t-statistic: ', t_statistic)
print('p-value: ', p_value)

print('male mean: ', male['score'].mean())
print('female mean: ', female['score'].mean())

t_statistic, p_value = ttest_ind(male['score'], female['score'],
                                 equal_var = True, alternative = 'less')
print('t-statistic: ', t_statistic)
print('p-value: ', p_value)

before = np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15])
after = np.array([10.52, 14.83, 13.03, 16.46, 10.84, 12.45])

t_statistic, p_value = ttest_rel(after, before, alternative = 'greater')
print('t-statistic: ', t_statistic)
print('p-value: ', p_value)

sample_d = after - before
t_statistic, p_value = ttest_1samp(sample_d, 0, alternative = 'greater')
print('t-statistic: ', t_statistic)
print('p-value: ', p_value)

result = ttest_ind(male['score'], female['score'], equal_var = False, alternative='greater')
print(result)

oj_lengths = np.array([17.6, 9.7, 16.5, 12.0, 21.5, 23.3, 23.6, 26.4, 20.0, 25.2,
                       25.8, 21.2, 14.5, 27.3, 23.8])
vc_lengths = np.array([7.6, 4.2, 10.0, 11.5, 7.3, 5.3, 5.8, 14.5, 10.6, 8.2, 9.4,
                      16.5, 9.7, 8.3, 13.6, 8.2])

s1 = oj_lengths.std(ddof = 1)
s2 = vc_lengths.std(ddof = 1)

ratio_of_variances = s1**2 / s2**2

print('ratio_of_variances: ', round(ratio_of_variances, 4))

def f_test(x, y, alternative = 'two_sided'):
    x = np.array(x)
    y = np.array(y)
    df1 = len(x) - 1
    df2 = len(y) - 1
    f_stat = np.var(x, ddof = 1) / np.var(y, ddof = 1)
    if alternative == 'greater':
        p = 1.0 - f.cdf(f_stat, df1, df2)
    elif alternative == 'less':
        p = f.cdf(f_stat, df1, df2)
    else:
        p = 1-f.cdf(f_stat, df1, df2)
        p = 2.0 * min(p, 1-p)
    return f_stat, p

f_value, p_value = f_test(oj_lengths, vc_lengths)

print('Test statistic: ', f_value)
print('p-value: ', p_value)