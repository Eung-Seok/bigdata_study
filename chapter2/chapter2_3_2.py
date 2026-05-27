import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 데이터 불러오기
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

# 원본 데이터의 컬럼별 결측치 개수 확인
print(df.isna().sum(axis = 0))

from sklearn.impute import SimpleImputer

# 평균값으로 goout 결측치 대체
train_X1 = train_X.copy()
test_X1 = test_X.copy()
imputer_mean = SimpleImputer(strategy = 'mean')

train_X1['goout'] = imputer_mean.fit_transform(train_X1[['goout']])
test_X1['goout'] = imputer_mean.transform(test_X[['goout']])

print(train_X1['goout'].isna().sum())
print(test_X1['goout'].isna().sum())

# 중앙값으로 goout 결측치 대체
train_X2 = train_X.copy()
test_X2 = test_X.copy()

imputer_median = SimpleImputer(strategy='median')
train_X2['goout'] = imputer_median.fit_transform(train_X2[['goout']])
test_X2['goout'] = imputer_median.transform(test_X2[['goout']])
print(train_X2['goout'].isna().sum())
print(test_X2['goout'].isna().sum())

# 최빈값으로 goout 결측치 대체
train_X3 = train_X.copy()
test_X3 = test_X.copy()

imputer_mode = SimpleImputer(strategy='most_frequent')
train_X3['goout'] = imputer_mode.fit_transform(train_X3[['goout']])
test_X3['goout'] = imputer_mode.transform(test_X3[['goout']])
print(train_X3['goout'].isna().sum())
print(test_X3['goout'].isna().sum())

from sklearn.impute import KNNImputer

# KNNImputer는 숫자형 데이터에만 적용하기 위해 숫자형/범주형 분리
train_X4 = train_X.copy()
test_X4 = test_X.copy()
train_X4_num  = train_X4.select_dtypes('number') 
test_X4_num = test_X4.select_dtypes('number')

train_X4_cat = train_X4.select_dtypes('object')
test_X4_cat = test_X4.select_dtypes('object')

knnimputer = KNNImputer(n_neighbors = 5)

# KNN 방식으로 숫자형 결측치 대체
train_X4_num_imputed = knnimputer.fit_transform(train_X4_num)
test_X4_num_imputed = knnimputer.transform(test_X4_num)

# KNNImputer 결과는 ndarray이므로 다시 DataFrame으로 변환
train_X4_num_imputed = pd.DataFrame(
    train_X4_num_imputed,
    columns = train_X4_num.columns,
    index= train_X4.index
)

test_X4_num_imputed = pd.DataFrame(
    test_X4_num_imputed,
    columns = test_X4_num.columns,
    index = test_X4.index
)

import sys
import sklearn
print(sys.executable)
print(sklearn.__version__)

# 범주형 컬럼과 결측치 대체된 숫자형 컬럼 다시 결합
train_X4 = pd.concat([train_X4_cat, train_X4_num_imputed], axis=1)
test_X4 = pd.concat([test_X4_cat, test_X4_num_imputed], axis = 1)

print(train_X4['goout'].isna().sum())
print(test_X4['goout'].isna().sum())

# set_output을 사용하면 imputer 결과를 pandas DataFrame으로 받을 수 있음
knnimputer2 = KNNImputer(n_neighbors= 5).set_output(transform='pandas')
train_X4_num_imputed2 = knnimputer2.fit_transform(train_X4_num)
test_X4_num_imputed2 = knnimputer2.transform(test_X4_num)

print(train_X4_num_imputed2.head())
print(train_X4_num_imputed2['goout'].isna().sum())
print(test_X4_num_imputed2['goout'].isna().sum())

# fillna 연습용 작은 예제 데이터
data = {
    '학생':['철수','영희','민수','수지','지현'],
    '수학':[85,np.nan,78,np.nan,93],
    '영어':[np.nan,88,79,85,np.nan],
    '과학':[92,85,np.nan,80,88]
}
df = pd.DataFrame(data)

# 각 과목 평균으로 결측치 대체
df1 = df.copy()
df1['수학'].fillna(df1['수학'].mean(), inplace = True)
df1['영어'].fillna(df1['영어'].mean(), inplace = True)
df1['과학'].fillna(df1['과학'].mean(),inplace = True)
print(df1)

# 특정 컬럼 결측치를 0으로 대체
df2= df.copy()
df2['수학'].fillna(0, inplace =True)
print(df2)

# 전체 결측치를 같은 값으로 대체
df2_1 = df.copy()
df2_1.fillna(50, inplace = True)
print(df2_1)

# 앞 행의 값으로 결측치 채우기
df2['영어'].fillna(method='ffill', inplace=True)
print(df2)

# 뒤 행의 값으로 결측치 채우기
df2['과학'].fillna(method='bfill', inplace=True)
print(df2)

# 다른 컬럼 값을 이용해서 결측치 채우기
df3 = df.copy()
df3['수학'].fillna(df3['영어'],inplace = True)
print(df3)

# 숫자형 컬럼만 골라 컬럼별 평균으로 결측치 대체
df4 =  df.copy()
df4_num = df4.select_dtypes('number')
df4_num = df4_num.apply(lambda col:col.fillna(col.mean()))
df4[df4_num.columns] = df4_num

print(df4)
