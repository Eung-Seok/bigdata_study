import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
#1 결측치를 중앙값으로 대치한 후 age의 평균을 계산하여 출력
data = {
    'name' : ['Alice', 'Bob', 'Charlie','David', 'Eva'],
    'age' : [25,np.nan,30,np.nan,22],
    'score' : [85,90,78,88,95]
}
df = pd.DataFrame(data)
df['age'] = SimpleImputer(strategy='median').fit_transform(df[['age']])
print(df['age'].mean())

#2 city변수를 원 핫 인코딩하여 새로운 데이터프레임으로 출력
data = pd.DataFrame({
    'name' : ['Alice', 'Bob', 'Charlie','David', 'Eva'],
    'city' : ['Seoul', 'Busan', 'Seoul', 'Daegu', 'Busan']
})

onehotencoder = OneHotEncoder(sparse_output= False).set_output(transform='pandas')
city_encoded = onehotencoder.fit_transform(data[['city']])
print(city_encoded)

#3 다음 데이터프레임에서 height 변수를 표준화한 후 해당 변수의 최대값과 최소값의 차이를 출력하시오
data = pd.DataFrame({
    'name' : ['Alice', 'Bob', 'Charlie','David', 'Eva'],
    'height' : [165,170,175,180,185]
})

scaler = StandardScaler()
data['height'] = scaler.fit_transform(data[['height']])
print(data['height'].max() - data['height'].min())

#4 다음 데이터에서 'city'변수의 결측치를 최빈값으로 대치한 후 각 도시별 인원수를 출력
data = pd.DataFrame({
    'name' : ['Alice', 'Bob', 'Charlie','David', 'Eva','Frank'],
    'city' : ['Seoul', np.nan, 'Busan', 'Seoul', np.nan, 'Busan']
})

data[['city']] = SimpleImputer(strategy='most_frequent').fit_transform(data[['city']])
print(data.groupby('city').count())
#print(data.groupby('city).value_counts()) 책에서 원한 정답

#5 다음 데이터에서 sales 변수의 이상치를 상자글미 기준으로 제거한 후 남은 데이터의 평균을 출력
data = pd.DataFrame({
    'product' : ['A','B','C','D','E','F','G'],
    'sales' : [100,120,130,400,110,115,500]
})
Q1 = data['sales'].quantile(0.25)
Q3 = data['sales'].quantile(0.75)
IQR = Q3-Q1
data = data[(data['sales'] >= Q1 - 1.5*IQR) & (data['sales'] <= Q3 + 1.5*IQR)]
print(data['sales'].mean())

#6 다음 데이터에서 score변수를 구간의 폭이 동일하도록 4개의 구간으로 이산화하고 각 구간의 데이터 개수를 출력
data = pd.DataFrame({
    'name' : ['A','B','C','D','E','F','G','H','I'],
    'score' : [55,60,65,70,75,80,85,90,95]
})

kid = KBinsDiscretizer(n_bins = 4, strategy='uniform',encode='ordinal')
data['score_bin'] = kid.fit_transform(data[['score']]).astype(int)
print(data['score_bin'].value_counts())

#7 다음 데이터에서 math english science 3개 변수를 대상으로 주성분 분석을 수행하여,
# 전체 분산의 80% 이상을 설명하는 최소 주성분 개수를 출력하시오
data = pd.DataFrame({
    'name' : ['Alice', 'Bob', 'Charlie','David', 'Eva'],
    'math' : [80,85,78,92,88],
    'english' : [75,90,58,95,92],
    'science' : [82,88,79,94,90]
})

X = data[['math', 'english', 'science']]
X_scaled = scaler.fit_transform(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=0.8,svd_solver='full')

X_pca = pca.fit_transform(X_scaled)
print(X_pca.shape[1])