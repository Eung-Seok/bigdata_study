import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from scipy.stats import anderson, norm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from scipy.stats import chisquare


data = np.array([155,126,27,82,115,140,73,92,110,134])

sorted_data = np.sort(data)

minimum = np.min(sorted_data)
maximum = np.max(sorted_data)

median = np.median(sorted_data)

lower_half = sorted_data[sorted_data < median]
upper_half = sorted_data[sorted_data > median]

q1 = np.median(lower_half)
q3 = np.median(upper_half)

print('최솟값: ', minimum)
print('최댓값: ', maximum)
print('중앙값: ', median)
print('Q1: ', q1)
print('Q3: ', q3)

iqr = q3 - q1
print('IQR: ', iqr)

sorted_data = np.sort(data)
n = len(data)

q3_percentile = (n-1) * 0.75 + 1
j3, h3 = divmod(q3_percentile, 1)
q3 = (1 - h3) * sorted_data[int(j3) - 1] + h3 * sorted_data[int(j3)]
print(q3)

x = np.array([155,126,27,82,115,140,73,92,110,134])

q1 = np.percentile(x, 25)
q2 = np.percentile(x, 50)
q3 = np.percentile(x, 75)
print('1사분위수: ', q1)
print('2사분위수: ',q2)
print('3사분위수: ',q3)

percentiles = np.array([sp.percentileofscore(x, value, kind='rank') for value in x])
print(percentiles)

data_x = np.array([4.62,4.09,6.2,8.24,0.77,5.55,3.11,11.97,2.16,3.24,10.91,11.36,0.87])
percentile_rank = np.array([sp.percentileofscore(data_x, value, kind='rank') for value in data_x])
print(percentile_rank[:6])

theory_x = sp.norm.ppf(percentile_rank/100, np.mean(data_x), np.std(data_x))
print(theory_x[:6])

# plt.scatter(theory_x, data_x, color='k')
# plt.plot([0,12], [0,12], 'k')
# plt.title('QQplot')
# plt.xlabel('Theoritical Quantiles')
# plt.ylabel('Sample Quantiles')
# plt.show()
# sp.probplot(data_x, dist = 'norm', plot = plt)
# plt.show()

w, p_value = sp.shapiro(data_x)
print('W: ', w)
print('p-value: ', p_value)

sample_data = np.array([4.62,4.09,6.2,8.24,0.77,5.55,3.11,11.97,2.16,3.24,10.91,11.36,0.87])

result = sp.anderson(sample_data, dist='norm')

print('검정통계량: ', result[0])
print('임곗값: ', result[1])
print('유의수준: ', result[2])

# ecdf = ECDF(sample_data)
# x = np.linspace(min(sample_data), max(sample_data))
# y = ecdf(x)

# plt.scatter(x,y)
# plt.title('Estimated CDF vs CDF')

# k = np.arange(min(sample_data), max(sample_data), 0.1)
# plt.plot(k, norm.cdf(k, loc = np.mean(sample_data), scale = np.std(sample_data, ddof = 1)), color = 'k')
# plt.show()

sample_data = [10.67, 9.92, 9.62, 9.53, 9.14, 9.74, 8.45,
               12.65, 11.47, 8.62]
n = len(sample_data)
sample_variance = np.var(sample_data, ddof = 1)
t = (n-1) * sample_variance / 1.3
print(t)

print('p-value: ', 1-chi2.cdf(t, df = n-1))

critical_value = chi2.ppf(0.95, df = 1)
print(critical_value)
pvalue = chi2.sf(15.55, df = 1)
print(pvalue)

table = np.array([[14,4],[0,10]])

chi2, p, df, expected = chi2_contingency(table, correction = False)
print('X-squared: ', chi2.round(3), 'df: ', 'p-value: ', p.round(3))
print(expected)

table = np.array([[50,30,20],[45,35,20]])
chi2, p, df, expected = chi2_contingency(table, correction = False)
print(chi2.round(3), p.round(3))
print(expected)

observed = np.array([13,23,24,20,27,18,15])
expected = np.repeat(20,7)

statistic, p_value = chisquare(observed, f_exp = expected)
print('Test statistic: ', statistic.round(3))
print('p-value: ', p_value.round(3))
print('Expected: ', expected)