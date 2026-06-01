import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.stats.mstats import brunnermunzel
from scipy.stats import levene
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import binom

sample = np.array([9.76,11.1,10.7,10.72,11.8,6.15,10.52,
                   14.83,13.03,16.46,10.84,12.45])

sample_diff = abs(np.array(sample) - 10)
r_i = rankdata(sample_diff)
psi_i = np.where(sample - 10 >= 0, 1, 0)

print(sum(r_i[psi_i > 0]))

eta_0 = 10
statistics, pvalue = wilcoxon(sample-eta_0, alternative = 'two-sided')

print('Test statistic: ', statistics)
print('p-value: ', pvalue)

a = np.arange(start = 1, stop = 13)
print(sum(a) - statistics)

gender = ['female'] * 7 + ['male'] * 5
data_mww = pd.DataFrame({'id': range(1,13), 'score': sample, 'gender': gender})

n1 = len(data_mww[data_mww['gender'] == 'female'])
n2 = len(data_mww) - n1

r_i = rankdata(data_mww['score'])
r_1p = sum(r_i[ : 7])
r_2p = sum(r_i[7 : ])

u1 = n1 * n2 + sum(range(1, n1 + 1)) - r_1p
u2 = n1 * n2 + sum(range(1, n2 + 1)) - r_2p
U = min(u1, u2)

print(U)

female = data_mww[data_mww['gender'] == 'female']['score']
male = data_mww[data_mww['gender'] == 'male']['score']

stat, pvalue = mannwhitneyu(female, male, method = 'exact')

print('stat: ', stat.round(3))
print('p-value: ', pvalue.round(3))

stat, pvalue = brunnermunzel(female, male, alternative = 'two-sided')

print('stat: ', stat.round(4))
print('p-value: ', pvalue.round(4))

id = [1,2,3,4,5,6]
before_after = ['before'] * 6 + ['after'] * 6
tab3 = pd.DataFrame({'id' : id*2, 'score' : sample, 'group' : before_after})

test3_data = tab3.pivot(index = 'id', columns = 'group', values = 'score')
test3_data['score_diff'] = test3_data['after'] - test3_data['before']

print(test3_data['score_diff'])

sample_sign = np.sign(test3_data['score_diff'])
print(sum(rankdata(abs(test3_data['score_diff']))[sample_sign > 0]))

statistics, pvalue = wilcoxon(test3_data['score_diff'], alternative = 'greater')

print('Test statistics: ', statistics)
print('p-value: ',pvalue)

mydata = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/tooth_growth.csv')
print(mydata.head())

a = mydata[mydata['supp'] == 'OJ']['len']
b = mydata[mydata['supp'] == 'VC']['len']

statistics, pvalue = levene(a, b, center = 'mean')
print('Test statistics: ', statistics)
print('p-value: ', pvalue)

sample_sign = np.sign(np.array(sample) - 10)
print(sum(sample_sign > 0))

statistics, pvalue = sign_test(sample, mu0 = 10)

print('Test statistics: ', statistics)
print('p-value: ', pvalue)

print((1 - binom.cdf(9,12,0.5)) * 2)