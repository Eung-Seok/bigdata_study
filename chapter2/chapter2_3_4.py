import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import warnings
import matplotlib.pyplot as plt
np.warnings = warnings
from sklearn.preprocessing import OneHotEncoder

# 학생 데이터 불러오기
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')

# 입력 변수 X와 타깃 변수 y 분리
X = df.drop(['grade'], axis=1)
y = df.grade

# school 비율을 유지해서 train/test 분리
train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=X['school'],
    random_state=0
)

# 자전거 대여량 데이터로 분포 변환 연습
bike_data = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bike_train.csv')
# bike_data['count'].hist();
# plt.show();

# Box-Cox 변환은 양수 데이터에만 적용 가능
box_tr = PowerTransformer(method='box-cox')
bike_data['count_boxcox'] = box_tr.fit_transform(bike_data[['count']])
print(box_tr.lambdas_)

# 로그 변환과 제곱근 변환
bike_data['count_log'] = np.log1p(bike_data[['count']])
bike_data['count_sqrt'] = np.sqrt(bike_data[['count']])
# bike_data[['count', 'count_boxcox', 'count_log', 'count_sqrt']].hist();
# plt.show();

from sklearn.preprocessing import StandardScaler

# StandardScaler로 숫자형 컬럼 표준화
train_X9 = train_X.copy()
test_X9 = test_X.copy()

train_X9_num = train_X9.select_dtypes('number')
test_X9_num = test_X9.select_dtypes('number')

stdscaler = StandardScaler().set_output(transform='pandas')
train_X9_num = stdscaler.fit_transform(train_X9_num)
test_X9_num = stdscaler.transform(test_X9_num)

# fig, axs = plt.subplots(nrows = 1, ncols=2)
# train_X9['absences'].hist(ax = axs[0],  color = 'blue', alpha = 0.7)
# axs[0].set_title('before transformation')

# train_X9_num['absences'].hist(ax = axs[1], color = 'red', alpha = 0.7)
# axs[1].set_title('after transformation')
# plt.tight_layout();
# plt.show();

# 표준화 전후 평균과 표준편차 확인
print(np.round(train_X9['absences'].mean()))
print(np.round(train_X9_num['absences'].mean()))
print(np.round(train_X9['absences'].std(),2))
print(np.round(train_X9_num['absences'].std(),2))

from sklearn.preprocessing import MinMaxScaler

# MinMaxScaler로 숫자형 컬럼을 0~1 범위로 변환
train_X10 = train_X.copy()
test_X10 = test_X.copy()

train_X10_num = train_X10.select_dtypes('number')
test_X10_num = test_X10.select_dtypes('number')

minmaxscaler = MinMaxScaler().set_output(transform='pandas')
train_X10_num = minmaxscaler.fit_transform(train_X10_num)
test_X10_num = minmaxscaler.transform(test_X10_num)

# 변환 후 각 컬럼의 범위가 1인지 확인
range_df = train_X10_num.select_dtypes('number').apply(lambda x: x.max() - x.min(), axis = 0)
print(range_df)

# IQR 기준 이상치 탐지
warpbreaks = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/warpbreaks.csv')
# warpbreaks.boxplot(column = ['breaks']);
# plt.show();

Q1 = np.quantile(warpbreaks['breaks'], 0.25)
Q3 = np.quantile(warpbreaks['breaks'],0.75)
IQR = Q3 - Q1

UC = Q3 + (1.5 * IQR)
LC = Q1 - (1.5 * IQR)
print(warpbreaks.loc[(warpbreaks.breaks > UC) | (warpbreaks.breaks < LC), :])

# warpbreaks.loc[(warpbreaks.breaks <= UC) & (warpbreaks.breaks >= LC), :]

# 평균 +- 3 표준편차 기준 이상치 탐지
upper = warpbreaks['breaks'].mean() + (3*warpbreaks['breaks'].std())
lower = warpbreaks['breaks'].mean() - (3*warpbreaks['breaks'].std())

print(warpbreaks.loc[(warpbreaks.breaks > upper) | (warpbreaks.breaks < lower), :].head(3))

from sklearn.preprocessing import KBinsDiscretizer

# 구간화 연습용 1차원 데이터
X = np.array([[0,1,1,2,5,10,11,14,18]]).T

# 동일한 너비로 3개 구간 생성
kbd = KBinsDiscretizer(n_bins = 3,
                       strategy='uniform')
X_bin = kbd.fit_transform(X).toarray()
print(kbd.bin_edges_)

# 분위수 기준으로 4개 구간 생성
kbd2 = KBinsDiscretizer(n_bins = 4, strategy='quantile')
X_bin2 = kbd2.fit_transform(X).toarray()
print(kbd2.bin_edges_)

# 실제 분위수 값 확인
print(np.quantile(X,[0.25,0.5,0.75,1]))

# pd.cut으로 직접 구간과 라벨 지정
bins = [0,4,7,11,18]
labels = ['A','B','C','D']
X_bin3 = pd.cut(X.reshape(-1),
                bins = bins,
                labels = labels)
print(X_bin3)

# PCA 연습용 10종 경기 데이터 불러오기
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/df_5.csv')
df.columns = ['index','X100m','Long.jump','Shot.put','High.jump','X400m','X110m.hurdle','Discus','Pole.vault','Javeline','X1500m']
df.set_index('index',inplace = True)
print(df.head())
print(df.shape)

from sklearn.model_selection import train_test_split

# PCA 적용 전 train/test 분리
train, test = train_test_split(df,test_size=0.3, random_state=42)

import seaborn as sns
import matplotlib.pyplot as plt

# 변수 간 상관관계 확인
corr_matrix = train.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix (Train Data)')
# plt.show()

# PCA 전에 모든 변수 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

from sklearn.decomposition import PCA

# 주성분 10개를 모두 계산
pca = PCA(n_components=10, svd_solver = 'auto')

X_train_pca = pca.fit_transform(train_scaled)
X_test_pca = pca.transform(test_scaled)

import matplotlib.pyplot as plt

# 누적 설명분산 비율 확인
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# plt.figure(figsize=(8,6))
# plt.plot(range(1, len(cumulative_explained_variance) + 1),
#          cumulative_explained_variance, marker='o', linestyle='-')
# plt.grid(True)
# plt.show()

# 누적 설명분산이 80% 이상이 되도록 주성분 개수 자동 선택
pca = PCA(n_components=0.8, svd_solver='full')

X_train_pca = pca.fit_transform(train_scaled)
X_test_pca = pca.transform(test_scaled)

print(pca.explained_variance_ratio_)
print(pca.n_components_)

from sklearn.compose import make_column_transformer

# ColumnTransformer 연습용 데이터 불러오기
dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bda1.csv')
y = dat.grade
X = dat.drop(['grade'], axis=1)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)

# 범주형 컬럼과 숫자형 컬럼 분리
cat_columns = train_X.select_dtypes('object').columns
num_columns = train_X.select_dtypes('number').columns

# 범주형은 원핫 인코딩, 숫자형은 표준화
onehotencoder = OneHotEncoder(sparse_output=False,
                              drop=None,
                              handle_unknown='ignore')

stdscaler = StandardScaler()

# 컬럼별로 다른 전처리를 한 번에 적용
mc_transformer = make_column_transformer(
    (onehotencoder, cat_columns),
    (stdscaler, num_columns),
    remainder = 'passthrough'
).set_output(transform='pandas')

# train에는 fit_transform, test에는 transform 적용
train_X_transformed = mc_transformer.fit_transform(train_X)
test_X_transformed = mc_transformer.transform(test_X)

print(train_X_transformed.head())

from sklearn.compose import ColumnTransformer

onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
stdscaler = StandardScaler()

c_transformer = ColumnTransformer(
    transformers = [
        ('cat', onehotencoder, cat_columns),
        ('num', stdscaler, num_columns)
    ]
).set_output(transform = 'pandas')

train_X2_transformed = c_transformer.fit_transform(train_X)
test_X2_transformed = c_transformer.transform(test_X)

print(train_X2_transformed.head())
