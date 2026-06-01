import numpy as np
import matplotlib.pyplot as plt

mu, sigma  = 30,5
sample_sizes = [10, 50, 100, 500, 1000, 5000]
num_means = 1000

# fig, axes = plt.subplots(2,3, figsize= (15,15))
# axes = axes.flatten()

# x_limits = (mu - 3*sigma, mu + 3*sigma)

# for i, size in enumerate(sample_sizes):
#     sample_means = [np.mean(np.random.normal(mu, sigma, size)) for _ in range(num_means)]
#     axes[i].hist(sample_means, bins = 30, alpha = 0.75, color = 'blue', edgecolor = 'black')
#     axes[i].axvline(mu, color = 'green', linestyle='dashed', linewidth=1)
#     axes[i].set_xlim(x_limits)
#     axes[i].set_title(f'Sample Size: {size}\nNumber of Means: {num_means}')
#     axes[i].set_xlabel('Mean Value')
#     axes[i].set_ylabel('Frequency')
#     axes[i].legend(['Population Mean'])
# plt.tight_layout()
# plt.show()

data = [4.3,4.1,5.2,4.9,5.0,4.5,4.7,4.8,5.2,4.6]

from scipy.stats import t

mean = np.mean(data)
n = len(data)
se = np.std(data, ddof = 1) / np.sqrt(n)
print(round(mean - t.ppf(0.975, n-1) * se, 3), round(mean + t.ppf(0.975, n-1) * se, 3))

ci = t.interval(0.95, loc = mean, scale = se, df = n-1)
print('95% 신뢰구간:', [round(i, 3) for i in ci])

from scipy.stats import norm
std = 5/np.sqrt(30)
print(norm.sf(83, loc = 80, scale = std))

x = [4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11, 11.97, 2.16, 3.24, 10.91, 11.36, 0.87, 9.93, 2.9]
t_value = (np.mean(x) - 7) / (np.std(x, ddof = 1) / np.sqrt(len(x)))
print(round(t_value, 3))