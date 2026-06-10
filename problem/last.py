# 빅분기 실기 기출형 종합 커버 세트

# 아래는 작업형 1 / 2 / 3 전체 패턴 커버용이다.

import pandas as pd
import numpy as np

np.random.seed(2026)
# 작업형 1
# 1-1
loan = pd.DataFrame({
    '연도': np.random.choice([2020, 2021, 2022, 2023], 180),
    '지역코드': np.random.choice(['A01','A02','A03','A04','A05','A06'], 180),
    '성별': np.random.choice(['남성','여성'], 180),
    '대출액': np.random.randint(100, 2000, 180).astype(float)
})

loan.loc[[3, 17, 44, 89, 120], '대출액'] = np.nan

# 대출액 결측치는 전체 중앙값으로 대체하시오.
# 2022년 데이터만 사용하여 지역코드별 남성 총 대출액과 여성 총 대출액의 차이가 가장 큰 지역코드를 출력하시오.

loan['대출액'] = loan['대출액'].fillna(loan['대출액'].median())
df  = loan[loan['연도'] == 2022]
df = df.groupby(['지역코드', '성별'])['대출액'].sum().reset_index()
df_pivot = pd.pivot_table(df, index = '지역코드', columns = '성별', values = '대출액', fill_value = 0)
print('A06')
# 1-2
crime = pd.DataFrame({
    '연도': np.random.choice([2020, 2021, 2022, 2023], 220),
    '범죄유형': np.random.choice(['절도','폭행','사기','방화','강도'], 220),
    '발생건수': np.random.randint(50, 700, 220),
    '검거건수': np.random.randint(10, 650, 220)
})

crime.loc[crime['검거건수'] > crime['발생건수'], '검거건수'] = crime['발생건수']

# 검거율을 검거건수 / 발생건수로 계산하시오.
# 각 연도별 검거율이 가장 높은 범죄유형을 찾고, 해당 행들의 검거건수 합계를 정수로 출력하시오.

crime['검거율'] = crime['검거건수']/crime['발생건수']
indexes = df.idxmax()
print(crime['검거율'].rank(ascending = False).head())

1-3
emp = pd.DataFrame({
    '사원ID': [f'E{i:03d}' for i in range(1, 121)],
    '부서': np.random.choice(['영업','개발','인사','재무','기획'], 120),
    '등급': np.random.choice(['A','B','C'], 120),
    '근속연수': np.random.randint(1, 21, 120).astype(float),
    '성과점수': np.random.randint(40, 101, 120)
})

emp.loc[[2, 9, 15, 22, 31, 40, 55, 63, 77, 101], '근속연수'] = np.nan

# 근속연수 결측치는 각 부서와 등급별 평균 근속연수로 채우시오.
# 단, 평균값의 소수점은 버림 처리하시오.
# 이후 성과점수가 80점 이상인 직원 중 근속연수 평균을 소수 둘째 자리까지 출력하시오.

emp['근속연수'] = emp['근속연수'].fillna(np.floor(emp.groupby(['부서', '등급'])['근속연수'].transform('mean')))
print(f"{emp[emp['성과점수'] >= 80]['근속연수'].mean():.2f}")


# 1-4
log = pd.DataFrame({
    'user_code': [f'user-{i:03d}' for i in range(1, 151)],
    '접속일': pd.date_range('2021-12-15', periods=150, freq='5D'),
    '메시지': np.random.choice([
        'login success',
        'payment failed retry',
        'user logout',
        'password reset request',
        'data upload success',
        'login failed retry'
    ], 150),
    '사용시간': np.random.randint(5, 400, 150)
})

# user_code에서 숫자 부분만 추출하여 정수형 user_num 컬럼을 만드시오.
# 접속일에서 연도와 월을 추출하시오.
# user_num이 30 이상 120 이하이고, 접속 연도가 2022년이며, 접속 월이 6월 이후인 데이터에서 사용시간의 제3사분위수를 정수로 출력하시오.

log['user_num'] = log['user_code'].str.extract(r'([0-9]+)').astype(int)
log['접속일'] = pd.to_datetime(log['접속일'])
log['연도'] = log['접속일'].dt.year
log['월'] = log['접속일'].dt.month
print(log[(log['user_num'] >= 30) & (log['user_num'] <= 120) & (log['연도'] == 2022) & (log['월'] >= 6) ]['사용시간'].head())
print(np.floor(log[(log['user_num'] >= 30) & (log['user_num'] <= 120) & (log['연도'] == 2022) & (log['월'] >= 6) ]['사용시간'].quantile(0.75)))

# 1-5
score = pd.DataFrame({
    '학생ID': [f'S{i:03d}' for i in range(1, 101)],
    '반': np.random.choice(['A','B','C','D'], 100),
    '성별': np.random.choice(['남','여'], 100),
    '국어': np.random.randint(40, 101, 100).astype(float),
    '수학': np.random.randint(35, 101, 100).astype(float),
    '영어': np.random.randint(30, 101, 100).astype(float)
})

score.loc[[4, 18, 37, 66], '수학'] = np.nan
score.loc[[9, 25, 71], '영어'] = np.nan

# 수학 결측치는 수학 중앙값으로 대체하시오.
# 영어 결측치가 있는 행은 제거하시오.
# 세 과목 평균점수를 만들고, 평균점수의 이상치 하한보다 작은 행의 개수를 출력하시오.
# 이상치 기준은 IQR 방식이다.

score['수학'] = score['수학'].fillna(score['수학'].median())
score = score.dropna(subset = ['영어'])
score['평균점수'] = score[['국어', '수학', '영어']].mean(axis = 1)
Q1 = score['평균점수'].quantile(0.25)
Q3 = score['평균점수'].quantile(0.75)
IQR = Q3 - Q1
anormal = Q1 - 1.5 * IQR
cnt = (score['평균점수'] < anormal).sum()
print(cnt)


# 1-6
sales = pd.DataFrame({
    '주문ID': [f'O{i:04d}' for i in range(1, 181)],
    '주문일': pd.date_range('2022-01-01', periods=180, freq='2D'),
    '지역': np.random.choice(['서울','부산','대구','광주','대전'], 180),
    '상품': np.random.choice(['A','B','C','D'], 180),
    '수량': np.random.randint(1, 20, 180),
    '가격': np.random.randint(1000, 50000, 180)
})

# 매출액 = 수량 * 가격 컬럼을 만드시오.
# 2022년 하반기 데이터만 사용하여 지역별 매출액 합계를 구하시오.
# 매출액 합계가 두 번째로 큰 지역명을 출력하시오.

sales['매출액'] = sales['수량'] * sales['가격']
sales['주문일'] = pd.to_datetime(sales['주문일'])
sales['연도'] = sales['주문일'].dt.year
sales['월'] = sales['주문일'].dt.month

sales_edit = sales[(sales['연도'] == 2022) & (sales['월']>=7)]
print(sales_edit.groupby('지역')['매출액'].sum()) 
print(sales_edit.groupby('지역')['매출액'].sum().sort_values(ascending = False).index[1])

# 1-7
member = pd.DataFrame({
    '회원ID': [f'M{i:03d}' for i in range(1, 131)],
    '가입일': pd.date_range('2021-01-01', periods=130, freq='9D'),
    '최종접속일': pd.date_range('2021-02-01', periods=130, freq='11D'),
    '등급': np.random.choice(['silver','gold','vip'], 130),
    '구매횟수': np.random.randint(0, 40, 130)
})

# 최종접속일 - 가입일의 차이를 일수로 계산하여 이용일수 컬럼을 만드시오.
# 등급별 이용일수 평균을 구하고, 평균 이용일수가 가장 큰 등급의 구매횟수 평균을 소수 둘째 자리까지 출력하시오.

member['최종접속일'] = pd.to_datetime(member['최종접속일'])
member['가입일'] = pd.to_datetime(member['가입일'])

member['이용일수'] = (member['최종접속일'] - member['가입일']).dt.days
idxmax = member.groupby('등급')['이용일수'].mean().idxmax()
print(f"{member.groupby('등급')['구매횟수'].mean()[idxmax]:.2f}")

# 1-8
health = pd.DataFrame({
    'id': [f'H{i:03d}' for i in range(1, 121)],
    '성별': np.random.choice(['M','F'], 120),
    '나이': np.random.randint(20, 70, 120),
    '키': np.random.normal(170, 8, 120),
    '몸무게': np.random.normal(70, 12, 120)
})

health.loc[[5, 19, 43], '키'] = np.nan

# 키 결측치는 성별별 평균 키로 대체하시오.
# BMI를 몸무게 / (키/100)**2로 계산하시오.
# 나이가 40 이상인 사람 중 BMI가 상위 10% 이상인 사람 수를 출력하시오.

health['키'] = health['키'].fillna(health.groupby('성별')['키'].transform('mean'))
health['BMI'] = health['몸무게'] / np.power(health['키']/100,2)
cnt = ((health['나이'] >= 40) & (health['BMI'] <= health['BMI'].quantile(0.1))).sum()
print(cnt)

# 1-9
review = pd.DataFrame({
    '리뷰ID': [f'R{i:03d}' for i in range(1, 111)],
    '상품군': np.random.choice(['전자','식품','의류','생활'], 110),
    '리뷰': np.random.choice([
        'good product',
        'very good product',
        'bad quality',
        'not bad',
        'delivery was very fast',
        'price is too high'
    ], 110),
    '평점': np.random.randint(1, 6, 110)
})

# 리뷰의 단어 수를 계산하여 word_count 컬럼을 만드시오.
# 상품군별 평균 단어 수를 구하고, 평균 단어 수가 가장 낮은 상품군의 평균 평점을 소수 둘째 자리까지 출력하시오.

review['word_count'] = review['리뷰'].str.split().str.len()
min = review.groupby('상품군')['word_count'].mean().idxmin()
print(f"{review.groupby('상품군')['평점'].mean()[min]:.2f}")

# 1-10
rank_df = pd.DataFrame({
    '선수ID': [f'P{i:03d}' for i in range(1, 101)],
    '팀': np.random.choice(['A','B','C','D','E'], 100),
    '득점': np.random.randint(0, 40, 100),
    '도움': np.random.randint(0, 15, 100),
    '출전시간': np.random.randint(10, 90, 100)
})

# 공격포인트 = 득점 + 도움 컬럼을 만드시오.
# 팀별 공격포인트 합계를 구하고, 합계 기준 dense rank를 내림차순으로 부여하시오.
# 2위 팀의 출전시간 평균을 소수 둘째 자리까지 출력하시오.

rank_df['공격포인트'] = rank_df['득점'] + rank_df['도움']
sum = rank_df.groupby('팀')['공격포인트'].sum().rank(ascending = False, method = 'dense')
print(f"{rank_df.groupby('팀')['출전시간'].mean()[sum.sort_values().index[1]]:.2f}")

# 1-11
raw = pd.DataFrame({
    'code': ['A-001', 'A-002', 'B-010', 'B-011', 'C-100', 'C-101'] * 20,
    '금액': np.random.randint(100, 10000, 120),
    '상태': np.random.choice(['정상', '취소', '보류'], 120),
    '분류': np.random.choice(['x', 'y', 'z'], 120)
})

# code에서 앞의 알파벳만 추출하여 group_code 컬럼을 만드시오.
# 상태가 정상인 데이터만 사용하여 group_code별 금액 평균을 구하시오.
# 평균 금액이 가장 큰 group_code를 출력하시오.

raw['group_code'] = raw['code'].str.extract(r'([a-zA-Z]+)')
normal = raw[raw['상태'] == '정상']
print(normal.groupby('group_code')['금액'].mean().idxmax())

# 1-12
scale_df = pd.DataFrame({
    'id': range(1, 101),
    'v1': np.random.randint(10, 500, 100),
    'v2': np.random.randint(100, 1000, 100),
    'v3': np.random.randint(1, 200, 100)
})

# v1, v2, v3에 MinMax 정규화를 적용하시오.
# 정규화된 v1, v2, v3의 합계를 score로 만들고, score가 2.0 이상인 행의 개수를 출력하시오.

from sklearn.preprocessing import MinMaxScaler
scale_df['v1'] = MinMaxScaler().fit_transform(scale_df[['v1']])
scale_df['v2'] = MinMaxScaler().fit_transform(scale_df[['v2']])
scale_df['v3'] = MinMaxScaler().fit_transform(scale_df[['v3']])
scale_df['score'] = scale_df[['v1', 'v2', 'v3']].sum(axis = 1)
cnt = (scale_df['score'] >= 2.0).sum()
print(cnt)


# 작업형 2
# 2-1 이진분류 AUC
from sklearn.datasets import make_classification

X_arr, y_arr = make_classification(
    n_samples=700,
    n_features=7,
    n_informative=5,
    n_redundant=1,
    n_classes=2,
    random_state=10
)

train_auc = pd.DataFrame(X_arr, columns=[f'x{i}' for i in range(1, 8)])
train_auc['gender'] = np.random.choice(['M','F'], 700)
train_auc['grade'] = np.random.choice(['A','B','C'], 700)
train_auc['cust_id'] = [f'C{i:04d}' for i in range(1, 701)]
train_auc['target'] = y_arr

X_test_arr, _ = make_classification(
    n_samples=150,
    n_features=7,
    n_informative=5,
    n_redundant=1,
    n_classes=2,
    random_state=11
)

test_auc = pd.DataFrame(X_test_arr, columns=[f'x{i}' for i in range(1, 8)])
test_auc['gender'] = np.random.choice(['M','F','U'], 150)
test_auc['grade'] = np.random.choice(['A','B','C','D'], 150)
test_auc['cust_id'] = [f'T{i:04d}' for i in range(1, 151)]

# train_auc를 이용하여 target을 예측하시오.
# 평가지표는 AUC이다.
# test_auc에 대한 예측 결과를 result.csv로 저장하시오.
# cust_id,pred

print(train_auc.shape, train_auc['cust_id'].nunique())
X = train_auc.drop(['target', 'cust_id'], axis = 1)
y = train_auc['target']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

num_columns = X.select_dtypes('number').columns.tolist()
cat_columns = X.select_dtypes('object').columns.tolist()

print(train_auc.head(5))
num_preprocess = make_pipeline(SimpleImputer(strategy = 'mean'))
cat_preprocess = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                               OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))

preprocess = ColumnTransformer([
    ('num', num_preprocess, num_columns),
    ('cat', cat_preprocess, cat_columns)
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('regressor', RandomForestClassifier(random_state = 42))
])

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.3, random_state = 42)

pipe.fit(train_X, train_y)
prob = pipe.predict_proba(valid_X)[:,1]
from sklearn.metrics import roc_auc_score
print('AUC: ', roc_auc_score(valid_y, prob))

pipe.fit(X, y)
cust_id = test_auc['cust_id']
test_X = test_auc.drop(['cust_id', 'target'], axis = 1, errors = 'ignore')
prob = pipe.predict_proba(test_X)[:, 1]
prob = pd.DataFrame({'cust_id': cust_id, 'pred': prob})
prob.to_csv('result.csv', index = False)



# 2-2 다중분류 Macro F1
X_arr, y_arr = make_classification(
    n_samples=600,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_classes=3,
    random_state=20
)

train_f1 = pd.DataFrame(X_arr, columns=[f'x{i}' for i in range(1, 7)])
train_f1['soil'] = np.random.choice(['A','B','C'], 600)
train_f1['region'] = np.random.choice(['north','south','east','west'], 600)
train_f1['ID'] = [f'TR{i:04d}' for i in range(1, 601)]
train_f1['라벨'] = y_arr

X_test_arr, _ = make_classification(
    n_samples=130,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_classes=3,
    random_state=21
)

test_f1 = pd.DataFrame(X_test_arr, columns=[f'x{i}' for i in range(1, 7)])
test_f1['soil'] = np.random.choice(['A','B','C','D'], 130)
test_f1['region'] = np.random.choice(['north','south','east','west','center'], 130)
test_f1['ID'] = [f'TE{i:04d}' for i in range(1, 131)]

# train_f1을 이용하여 라벨을 예측하시오.
# 평가지표는 Macro F1이다.
# test_f1에 대한 예측 결과를 result.csv로 저장하시오.
# ID,pred

print(train_f1.head())
X = train_f1.drop(['ID', '라벨'], axis = 1)
y = train_f1['라벨']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

num_columns = X.select_dtypes('number').columns.tolist()
cat_columns = X.select_dtypes('object').columns.tolist()

num_preprocess = make_pipeline(SimpleImputer(strategy = 'mean'))
cat_preprocess = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                               OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))

preprocess = ColumnTransformer([
    ('num', num_preprocess, num_columns),
    ('cat', cat_preprocess, cat_columns)
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('classifier', RandomForestClassifier(random_state = 42))
])

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)
pipe.fit(train_X, train_y)
pred = pipe.predict(valid_X)
print('f1-macro: ', f1_score(valid_y, pred, average = 'macro'))
pipe.fit(X, y)
id = test_f1['ID']
test_X = test_f1.drop(['ID', 'target'], axis = 1, errors = 'ignore')
pred = pipe.predict(test_X)
pred = pd.DataFrame({'ID': id, 'pred' : pred})
pred.to_csv('result.csv', index = False)


# 2-3 회귀 RMSE
from sklearn.datasets import make_regression

X_arr, y_arr = make_regression(
    n_samples=650,
    n_features=6,
    noise=15,
    random_state=30
)

train_rmse = pd.DataFrame(X_arr, columns=[f'x{i}' for i in range(1, 7)])
train_rmse['type'] = np.random.choice(['A','B','C'], 650)
train_rmse['area'] = np.random.choice(['S','M','L'], 650)
train_rmse['id'] = [f'R{i:04d}' for i in range(1, 651)]
train_rmse['price'] = y_arr + 500

X_test_arr, _ = make_regression(
    n_samples=140,
    n_features=6,
    noise=15,
    random_state=31
)

test_rmse = pd.DataFrame(X_test_arr, columns=[f'x{i}' for i in range(1, 7)])
test_rmse['type'] = np.random.choice(['A','B','C','D'], 140)
test_rmse['area'] = np.random.choice(['S','M','L','XL'], 140)
test_rmse['id'] = [f'RT{i:04d}' for i in range(1, 141)]

# train_rmse를 이용하여 price를 예측하시오.
# 평가지표는 RMSE이다.
# test_rmse에 대한 예측 결과를 result.csv로 저장하시오.
# id,pred

print(train_rmse.head())
X = train_rmse.drop(['id', 'price'], axis = 1,)
y = train_rmse['price']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

num_columns = X.select_dtypes('number').columns.tolist()
cat_columns = X.select_dtypes('object').columns.tolist()

num_preprocess = make_pipeline(SimpleImputer(strategy = 'mean'))
cat_preprocess = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                               OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))

preprocess = ColumnTransformer([
    ('num', num_preprocess, num_columns),
    ('cat', cat_preprocess, cat_columns)
])
pipe = Pipeline([
    ('preprocess', preprocess),
    ('regressor', RandomForestRegressor(random_state = 42))
])

train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 42, test_size = 0.3)
pipe.fit(train_X, train_y)
pred = pipe.predict(valid_X)
print("RMSE: ", np.sqrt(mean_squared_error(valid_y, pred)))

pipe.fit(X, y)
test_X = test_rmse.drop(['price', 'id'], axis = 1, errors = 'ignore')
id = test_rmse['id']
pred = pipe.predict(test_X)
pred = pd.DataFrame({'id': id, 'pred' : pred})
pred.to_csv('result.csv', index = False)

# 2-4 이진분류 F1
X_arr, y_arr = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    weights=[0.65, 0.35],
    random_state=40
)

train_bin_f1 = pd.DataFrame(X_arr, columns=[f'x{i}' for i in range(1, 6)])
train_bin_f1['job'] = np.random.choice(['A','B','C'], 500)
train_bin_f1['ID'] = [f'B{i:04d}' for i in range(1, 501)]
train_bin_f1['target'] = y_arr

X_test_arr, _ = make_classification(
    n_samples=120,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=41
)

test_bin_f1 = pd.DataFrame(X_test_arr, columns=[f'x{i}' for i in range(1, 6)])
test_bin_f1['job'] = np.random.choice(['A','B','C','D'], 120)
test_bin_f1['ID'] = [f'BT{i:04d}' for i in range(1, 121)]

# train_bin_f1을 이용하여 target을 예측하시오.
# 평가지표는 F1-score이다.
# test_bin_f1에 대한 예측 결과를 result.csv로 저장하시오.
# ID,pred

print(train_bin_f1.head())
X = train_bin_f1.drop(['ID', 'target'], axis = 1)
y = train_bin_f1['target']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

num_columns = X.select_dtypes('number').columns.tolist()
cat_columns = X.select_dtypes('object').columns.tolist()

num_preprocess = make_pipeline(SimpleImputer(strategy = 'mean'))
cat_preprocess = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                               OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))

preprocess = ColumnTransformer([
    ('num', num_preprocess, num_columns),
    ('cat', cat_preprocess, cat_columns)
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('classifier', RandomForestClassifier(random_state = 42))
])

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

pipe.fit(train_X, train_y)
pred = pipe.predict(valid_X)
print('f1-score: ',f1_score(valid_y, pred))
pipe.fit(X, y)
test_id = test_bin_f1['ID']
test_X = test_bin_f1.drop(['ID', 'target'], axis = 1, errors = 'ignore')
pred = pipe.predict(test_X)
pred = pd.DataFrame({'ID' : test_id, 'pred' : pred})
pred.to_csv('result.csv', index = False)

# 작업형 3
# 3-1
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

np.random.seed(100)

df31 = pd.DataFrame({
    'group': np.repeat([1, 2], 40),
    'value': np.r_[np.random.lognormal(3.0, 0.4, 40),
                   np.random.lognormal(3.2, 0.5, 40)]
})

# value에 자연로그를 적용하시오.
# 두 그룹의 분산비를 소수 셋째 자리까지 출력하시오. 큰 분산을 작은 분산으로 나눈다.
# 두 그룹의 합동분산을 소수 셋째 자리까지 출력하시오.
# 두 그룹 평균 차이에 대한 양측 검정의 p-value를 소수 셋째 자리까지 출력하시오. 등분산을 가정한다.

df31['value'] = np.log(df31['value'])
first = df31[df31['group'] == 1]
second = df31[df31['group'] == 2]
del min
f_var = first['value'].var()
s_var = second['value'].var()
print(f_var)
print(s_var)
ratio = max(f_var, s_var)/min(f_var, s_var)
print(f"{ratio:.3f}")
f_len = len(first['value'])
s_len = len(second['value'])
var = ((f_len - 1) * f_var + (s_len - 1) * s_var)/(f_len + s_len - 2)
print(f"{var:.3f}")
from scipy import stats
stat, pvalue = stats.ttest_ind(first['value'], second['value'], equal_var = True)
print(f"{pvalue:.3f}")
# 3-2
np.random.seed(101)

df32 = pd.DataFrame({
    'hours': np.random.randint(1, 12, 80)
})

df32['score'] = 50 + 4.2 * df32['hours'] + np.random.normal(0, 8, 80)
# hours와 score의 Pearson 상관계수를 소수 셋째 자리까지 출력하시오.
# 해당 상관계수 검정의 p-value를 소수 셋째 자리까지 출력하시오.

stat, pvalue = stats.pearsonr(df32['hours'], df32['score'])
print(f"{stat:.3f}")
print(f"{pvalue:.3f}")

# 3-3
np.random.seed(102)

df33 = pd.DataFrame({
    'id': [f'id{i}' for i in range(1, 201)],
    'f1': np.random.normal(50, 10, 200),
    'f2': np.random.normal(100, 15, 200),
    'f3': np.random.normal(30, 5, 200),
    'f4': np.random.normal(10, 2, 200),
    'f5': np.random.normal(0, 1, 200)
})

df33['design'] = (
    15
    + 0.5 * df33['f1']
    - 0.2 * df33['f2']
    + 2.0 * df33['f4']
    + np.random.normal(0, 5, 200)
)

# design을 종속변수로 하는 다중회귀분석을 수행하시오.
# 불필요한 컬럼은 제외하고, 절편항은 포함하시오.
# 훈련 데이터는 id1부터 id140까지이고, 테스트 데이터는 id141부터 id200까지이다.
# 훈련 데이터로 적합한 회귀모형에서 유의확률이 0.05 이상인 항의 개수를 출력하시오. 절편항도 포함한다.
# 훈련 데이터의 예측값과 실제값의 Pearson 상관계수를 소수 셋째 자리까지 출력하시오.
# 테스트 데이터의 RMSE를 소수 셋째 자리까지 출력하시오.

import statsmodels.api as sm
print(df33.info())
df33['num'] = df33['id'].index+1
train = df33[df33['num'] <= 140]
test = df33[df33['num'] > 140]

train_X = train.drop(['id', 'design'], axis = 1)
train_y = train['design']

model = sm.OLS(train_y, train_X).fit()
cnt = (model.pvalues >= 0.05).sum()
print(cnt)
pvalues = model.pvalues
sig_var = pvalues[pvalues < 0.05].index
train_X = train_X[list(sig_var)]
train_y = train['design']
model = sm.OLS(train_y, train_X).fit()

test_X = test[list(sig_var)]
test_y = test['design']
pred = model.predict(test_X)
stat, pvalue = stats.pearsonr(test_y, pred)
print(f"{stat:.3f}")
print(f"{np.sqrt(mean_squared_error(test_y, pred)):.3f}")

# 3-4
np.random.seed(103)

df34 = pd.DataFrame({
    'age': np.random.randint(20, 65, 300),
    'income': np.random.randint(1800, 7000, 300),
    'visit': np.random.randint(0, 12, 300),
    'delay': np.random.randint(0, 6, 300)
})

score = (
    -5
    + 0.04 * df34['age']
    + 0.0005 * df34['income']
    + 0.25 * df34['visit']
    - 0.55 * df34['delay']
)

prob = 1 / (1 + np.exp(-score))
df34['target'] = np.random.binomial(1, prob)

# target을 종속변수로 하는 로지스틱 회귀모형을 적합하시오.
# 절편항은 포함한다.
# delay 변수의 p-value를 소수 셋째 자리까지 출력하시오.
# visit이 1 증가할 때 target=1의 오즈비를 소수 셋째 자리까지 출력하시오.
# 전체 데이터에 대해 target=1 예측확률이 0.7 이상인 행의 개수를 출력하시오.

X = df34.drop(columns = 'target', axis = 1)
y = df34['target']
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
print(f"{model.pvalues['delay']:.3f}")
print(f"{np.exp(model.params['visit']):.3f}")

pred=  model.predict(X)
cnt = (pred >= 0.7).sum()
print(cnt)

# 3-5
np.random.seed(104)

df35 = pd.DataFrame({
    'x1': np.random.normal(10, 2, 120),
    'x2': np.random.normal(20, 5, 120),
    'x3': np.random.normal(30, 7, 120)
})

df35['y'] = 5 + 1.5 * df35['x1'] - 0.8 * df35['x2'] + np.random.normal(0, 3, 120)

# y를 종속변수로 하고 x1, x2, x3를 설명변수로 하는 다중회귀분석을 수행하시오. 절편항은 포함한다.
# 유의확률이 0.05 미만인 설명변수의 회귀계수 합을 소수 셋째 자리까지 출력하시오. 절편항은 제외한다.
# 수정 결정계수를 소수 셋째 자리까지 출력하시오.
# x1=12, x2=18, x3=35일 때 예측값을 소수 셋째 자리까지 출력하시오.

X = df35[['x1','x2','x3']]
y = df35['y']
X = sm.add_constant(X)
model = sm.OLS(y,X).fit()

pvalues = model.pvalues.drop('const')
print(f"{model.params.drop('const')[pvalues < 0.05].sum():.3f}")
print(f"{model.rsquared_adj:.3f}")
new_data = pd.DataFrame({'x1' : [12], 'x2' : [18], 'x3' : [35]})
new_data = sm.add_constant(new_data, has_constant = 'add')
print(f"{model.predict(new_data).iloc[0]:.3f}")

# 3-6
np.random.seed(105)

df36 = pd.DataFrame({
    'class': np.repeat(['A', 'B', 'C'], 30),
    'score': np.r_[np.random.normal(70, 8, 30),
                   np.random.normal(75, 7, 30),
                   np.random.normal(80, 9, 30)]
})

# 세 집단의 평균 차이를 검정하시오
# 검정통계량을 소수 셋째 자리까지 출력하시오..
# p-value를 소수 셋째 자리까지 출력하시오.

first = df36[df36['class'] == 'A']
second = df36[df36['class'] == 'B']
third = df36[df36['class'] == 'C']
from scipy import stats
stat, pvalue = stats.f_oneway(first['score'], second['score'], third['score'])
print(f"{stat:.3f}")
print(f"{pvalue:.3f}")

# 3-7
np.random.seed(106)

df37 = pd.DataFrame({
    'before': np.random.normal(80, 10, 60)
})

df37['after'] = df37['before'] + np.random.normal(3, 6, 60)

# 동일 대상의 전후 차이를 검정하시오.

# 검정통계량을 소수 셋째 자리까지 출력하시오.
# p-value를 소수 셋째 자리까지 출력하시오.

stat, pvalue = stats.ttest_rel(df37['after'], df37['before'])
print(f"{stat:.3f}")
print(f"{pvalue:.3f}")


# 3-8
np.random.seed(107)

df38 = pd.DataFrame({
    'gender': np.random.choice(['M','F'], 200),
    'buy': np.random.choice(['Y','N'], 200, p=[0.45, 0.55])
})

# gender와 buy의 독립성을 검정하시오.
# 검정통계량을 소수 셋째 자리까지 출력하시오.
# p-value를 소수 셋째 자리까지 출력하시오.

my = ((df38['gender']  == 'M') & (df38['buy'] == 'Y')).sum()
mn = ((df38['gender']  == 'M') & (df38['buy'] == 'N')).sum()
fy = ((df38['gender']  == 'F') & (df38['buy'] == 'Y')).sum()
fn = ((df38['gender']  == 'F') & (df38['buy'] == 'N')).sum()

df = pd.DataFrame([[my, mn], [fy, fn]])
print(df)
stat, pvalue,df, expected = stats.chi2_contingency(df)

print(stat)
print(pvalue)

# 3-9
np.random.seed(108)

df39 = pd.DataFrame({
    'x': np.random.normal(100, 15, 100)
})

# x가 평균 100, 표준편차 15인 정규분포를 따른다고 볼 수 있는지 검정하시오.
# 검정통계량을 소수 셋째 자리까지 출력하시오.
# p-value를 소수 셋째 자리까지 출력하시오.

df39 = (df39-100)/15
stat, pvalue = stats.kstest(df39['x'], 'norm', args = (0, 1))
print(f"{stat:.3f}")
print(f"{pvalue:.3f}")

# 3-10
np.random.seed(109)

df310 = pd.DataFrame({
    'actual': np.random.normal(100, 20, 80)
})

df310['pred'] = df310['actual'] + np.random.normal(0, 10, 80)
# RMSE를 소수 셋째 자리까지 출력하시오.
# MAE를 소수 셋째 자리까지 출력하시오.
from sklearn.metrics import mean_absolute_error
print(f"{np.sqrt(mean_squared_error(df310['actual'], df310['pred'])):.3f}")
print(f"{mean_absolute_error(df310['actual'], df310['pred']):.3f}")

# 3-11
np.random.seed(110)

df311 = pd.DataFrame({
    'id': [f'A{i:03d}' for i in range(1, 151)],
    'x1': np.random.normal(5, 1, 150),
    'x2': np.random.normal(10, 2, 150),
    'x3': np.random.normal(20, 3, 150)
})

df311['target'] = (
    3
    + 2.5 * df311['x1']
    - 1.2 * df311['x2']
    + np.random.normal(0, 2, 150)
)

# target을 종속변수로 다중회귀분석을 수행하시오.
# 불필요한 컬럼은 제외하고 절편항은 포함한다.
# 결정계수를 소수 셋째 자리까지 출력하시오.
# x2의 회귀계수를 소수 셋째 자리까지 출력하시오.
# x1=6, x2=9, x3=21일 때 예측값을 소수 셋째 자리까지 출력하시오.

X = df311.drop(['id', 'target'], axis = 1)
y = df311['target']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(f"{model.rsquared:.3f}")
print(f"{model.params['x2']:.3f}")
new_data = pd.DataFrame({'x1' : [6], 'x2' : [9], 'x3' : [21]})
new_data = sm.add_constant(new_data, has_constant = 'add')
print(f"{model.predict(new_data).iloc[0]:.3f}")

# 3-12
np.random.seed(111)

df312 = pd.DataFrame({
    'age': np.random.randint(20, 70, 250),
    'score': np.random.randint(300, 900, 250),
    'count': np.random.randint(0, 10, 250)
})

logit_score = -6 + 0.03 * df312['age'] + 0.006 * df312['score'] + 0.2 * df312['count']
prob = 1 / (1 + np.exp(-logit_score))
df312['target'] = np.random.binomial(1, prob)

# target을 종속변수로 하는 로지스틱 회귀모형을 적합하시오. 절편항은 포함한다.
# score의 회귀계수를 소수 셋째 자리까지 출력하시오.
# age가 1 증가할 때 target=0의 오즈비를 소수 셋째 자리까지 출력하시오.
# age=40, score=650, count=3일 때 target=1 확률을 소수 셋째 자리까지 출력하시오.

X = df312.drop(columns = 'target')
y = df312['target']
X = sm.add_constant(X)

model = sm.Logit(y, X).fit()
print(f"{model.params['score']:.3f}")
print(f"{np.exp(-model.params['age']):.3f}")
new_data = pd.DataFrame({'age' : [40], 'score': [650], 'count': [3]})
new_data = sm.add_constant(new_data, has_constant = 'add')
print(f"{model.predict(new_data).iloc[0]:.3f}")