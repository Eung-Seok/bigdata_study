import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import pearsonr

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

