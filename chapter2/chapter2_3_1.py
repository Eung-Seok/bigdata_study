import numpy as np 
import pandas as pd

# 데이터 불러오기
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')

# 입력 변수 X와 타깃 변수 y 분리
X = df.drop(['grade'], axis=1)
y = df.grade

from sklearn.model_selection import train_test_split

# 기본 train/test 분리
train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    test_size= 0.2,
    random_state=0,
    shuffle = True,
    stratify = None
)

# 분리된 데이터 크기 확인
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

# school 비율을 유지하도록 층화 추출
train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=X['school'],
    random_state=0
)

import matplotlib.pyplot as plt

# train/test 타깃 분포 비교
fig, axs = plt.subplots(nrows = 1, ncols = 2)
train_y.hist(ax = axs[0], color='blue', alpha = 0.7)
axs[0].set_title('histogram of train y')

test_y.hist(ax = axs[1], color='red', alpha=0.7)
axs[1].set_title('histogram of test y')
plt.tight_layout();
plt.show()
