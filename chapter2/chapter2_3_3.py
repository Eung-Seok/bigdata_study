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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

# OrdinalEncoder 연습용 데이터 복사
train_X6  = train_X.copy()
test_X6 = test_X.copy()

# 문자형 컬럼만 선택
train_X6_cat = train_X6.select_dtypes('object')
test_X6_cat = test_X6.select_dtypes('object')

# 범주형 값을 순서 없는 숫자 코드로 변환
ordinalencoder = OrdinalEncoder().set_output(transform='pandas')
train_X6_cat = ordinalencoder.fit_transform(train_X6_cat)
test_X6_cat = ordinalencoder.transform(test_X6_cat)

print(train_X6_cat.head(2))

# train에 없던 새로운 범주가 test에 등장하는 상황 예제
train_data = pd.DataFrame({'job': ['Doctor','Engineer','Teacher','Nurse']})

test_data = pd.DataFrame({'job' : ['Doctor','Lawyer','Teacher','Scientist']})

oe = OrdinalEncoder()

# 기본 OrdinalEncoder는 test에 새 범주가 있으면 transform에서 오류 발생
# train_data['job_encoded'] = oe.fit_transform(train_data[['job']])
# test_data['job_encoded'] = oe.transform(test_data[['job']])

# 새 범주를 -1로 처리하도록 옵션 지정
oe = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value= -1)

oe.fit(train_data[['job']])
OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=-1)

train_data['job_encoded'] = oe.transform(train_data[['job']])

test_data['job_encoded'] = oe.transform(test_data[['job']])
print(train_data)
print(test_data)

from sklearn.preprocessing import OneHotEncoder

# OneHotEncoder 연습용 데이터 복사
train_X7 = train_X.copy()
test_X7 = test_X.copy()

# 문자형 컬럼만 선택
train_X7_cat = train_X7.select_dtypes('object')
test_X7_cat = test_X7.select_dtypes('object')

# 원핫 인코딩: 새 범주는 무시하고, 결과는 pandas DataFrame으로 반환
onehotencoder = OneHotEncoder(sparse_output = False,
                              handle_unknown='ignore').set_output(transform='pandas')

train_X7_cat = onehotencoder.fit_transform(train_X7_cat)
test_X7_cat = onehotencoder.transform(test_X7_cat)

print(train_X7_cat.head())

# drop='first'로 첫 번째 더미 컬럼을 제거한 원핫 인코딩
train_X8 = train_X.copy()
test_X8 = test_X.copy()

train_X8_cat = train_X8.select_dtypes('object')
test_X8_cat = train_X8.select_dtypes('object')

dummyencoder = OneHotEncoder(sparse_output = False,
                             drop='first',
                             handle_unknown='error').set_output(transform='pandas')

train_X8_cat = dummyencoder.fit_transform(train_X8_cat)
test_X8_cat = dummyencoder.transform(test_X8_cat)
print(train_X8_cat.head())

# 희소 범주를 other로 묶는 예제
train_bike = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bike_train.csv')
test_bike = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bike_test.csv')

print(train_bike.head(2))

# weather 범주별 빈도 확인
print(train_bike['weather'].value_counts())

# weather 범주별 비율 계산
freq = train_bike['weather'].value_counts(normalize = True)
print(freq)

# 비율이 10% 미만인 희소 범주 선택
rare_categories = freq[freq < 0.1].index
print(rare_categories)

# train 기준 희소 범주를 other로 대체
train_bike['weather'] = train_bike['weather'].mask(train_bike['weather'].isin(rare_categories),'other')

# test에도 train에서 정한 같은 기준을 적용
test_bike['weather'] = test_bike['weather'].mask(test_bike['weather'].isin(rare_categories), 'other')
print(train_bike['weather'].value_counts())
