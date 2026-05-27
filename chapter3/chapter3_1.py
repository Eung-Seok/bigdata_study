import pandas as pd
import numpy as np

# 회귀 문제용 train/test 데이터 불러오기
train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/s11_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/s11_test.csv')

# 입력 변수와 타깃 변수 분리
train_X = train.drop(['grade'], axis=1)
train_y = train['grade']
test_X = test.drop(['grade'], axis=1)
test_y = test['grade']

from sklearn.model_selection import train_test_split

# train 데이터 안에서 검증용 데이터 한 번 더 분리
train_X_sub, valid_X, train_y_sub, valid_y = train_test_split(train_X, train_y, test_size = 0.3, random_state = 1)
print(train_X_sub.shape, train_y_sub.shape, valid_X.shape, valid_y.shape)

# 선형회귀 모델 학습
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_X_sub, train_y_sub)

# 검증 데이터 RMSE 계산
from sklearn.metrics import mean_squared_error
pred_val = lr.predict(valid_X)
print('valid RMSE: ', mean_squared_error(valid_y, pred_val, squared = False))

# 기본 교차검증으로 RMSE 확인
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(lr, train_X, train_y, scoring = 'neg_root_mean_squared_error')
rmse_score = -cv_score
mean_rmse_score = np.mean(rmse_score)

print('폴드별 RMSE: ', rmse_score)
print('교차검증 RMSE: ', mean_rmse_score)

from sklearn.model_selection import KFold

# KFold 옵션을 직접 지정한 교차검증
cv = KFold(n_splits= 5, shuffle = True, random_state = 0)
from sklearn.model_selection import cross_val_score
cv_score2 = cross_val_score(lr, train_X, train_y, scoring = 'neg_root_mean_squared_error', cv = cv)
rmse_score2 = -cv_score2
mean_rmse_score2 = np.mean(rmse_score2)
print('교차검증 RMSE: ', mean_rmse_score2)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# 랜덤포레스트 회귀 모델 생성
rf = RandomForestRegressor(random_state = 1)

# 모델이 가진 초매개변수 목록 확인
params = rf.get_params()

for param_name, param_value, in params.items():
    print(f"{param_name}: {param_value}")

# 튜닝할 초매개변수 후보 설정
param_grid = {'max_depth' : [10,20,30],
              'ccp_alpha' : [0.1,0.3,0.5]}

# GridSearchCV로 랜덤포레스트 초매개변수 탐색
rf_search = GridSearchCV(estimator= rf,
                         param_grid = param_grid,
                         cv = 5,
                         scoring = 'neg_root_mean_squared_error')
rf_search.fit(train_X,train_y)
best_params = rf_search.best_params_
print('최족의 초매개변수 조합', best_params)

mean_rmse_score3 = -rf_search.best_score_
print('교차검증 RMSE: ', mean_rmse_score3)

#데이터 누수얘기하면서 갑자기 나옴

# train에서 학습한 스케일러를 valid/test에 적용하는 예제
from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()

num_columns = train_X.select_dtypes('number').columns
train_X_numeric_scaled = stdscaler.fit_transform(train_X[num_columns])
valid_X_numeric_scaled = stdscaler.transform(valid_X[num_columns])
test_X_numeric_scaled = stdscaler.transform(test_X[num_columns])

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# 전처리와 SVR 모델을 Pipeline으로 연결
svr_pipe = Pipeline([
    ('preprocess', StandardScaler()),
    ('regressor', SVR())
])

svr_pipe.fit(train_X,train_y)

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

# make_pipeline을 사용한 같은 구조의 파이프라인
svr_pipe2 = make_pipeline(
    StandardScaler(),
    SVR()
)
svr_pipe2.fit(train_X,train_y)

from sklearn.model_selection import cross_val_score

# Pipeline 전체를 교차검증
cv_score4 = cross_val_score(svr_pipe,
                            train_X,
                            train_y,
                            scoring = 'neg_root_mean_squared_error',
                            cv = 5)

rmse_score4 = -cv_score4
mean_rmse_score4 = np.mean(rmse_score4)
print('교차검증 RMSE: ', mean_rmse_score4)

print('SVR 초매개변수: ', SVR().get_params())

# Pipeline 안의 regressor 단계 초매개변수는 regressor__C처럼 접근
SVR_param = {'regressor__C' : np.arange(1,100,20)}

from sklearn.model_selection import GridSearchCV

# SVR의 C 값을 GridSearchCV로 탐색
SVR_search = GridSearchCV(estimator = svr_pipe,
                          param_grid = SVR_param,
                          cv = 5,
                          scoring='neg_root_mean_squared_error')
SVR_search.fit(train_X, train_y)
print('Best 파라미터 조합: ', SVR_search.best_params_)
print('교차검증 RMSE: ', -SVR_search.best_score_)

# 숫자형/범주형 컬럼 목록 분리
num_columns = train_X.select_dtypes('number').columns.tolist()
cat_columns = train_X.select_dtypes('object').columns.tolist()

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 범주형 전처리: 원핫 인코딩
cat_preprocess = make_pipeline(
    OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)
)

# 숫자형 전처리: 평균 결측치 대체 후 표준화
num_preprocess = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

# num_preprocess = make_pipeline(
#     [('num', num_preprocess, num_columns),
#      ('cat', cat_preprocess, cat_columns)]
# )

from sklearn.compose import ColumnTransformer

# 컬럼 종류별로 다른 전처리 적용
preprocess = ColumnTransformer(
    [
        ('num', num_preprocess, num_columns),
        ('cat', cat_preprocess, cat_columns)
    ]
)

from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

# 전체 전처리와 SVR 모델을 하나의 Pipeline으로 구성
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('regressor', SVR())
    ]
)

SVR_param = {'regressor__C' : np.arange(1,100,20)}

# 여러 초매개변수를 동시에 탐색할 때 사용할 수 있는 형태
# SVR_param = {
#     'regressor__C' : [0.1,1,10],
#     'regressor__gamma' : [1e-3, 1e-4.'scale'],
#     'regressor__kernal' : ['linear','rbf']
# }

from sklearn.model_selection import GridSearchCV

# 전체 파이프라인 기준 GridSearchCV
SVR_search = GridSearchCV(estimator = full_pipe,
                          param_grid = SVR_param,
                          cv = 5,
                          scoring = 'neg_root_mean_squared_error')
SVR_search.fit(train_X, train_y)
print('Best 파라미터 조합: ', SVR_search.best_params_)
print('교차검증 RMSE: ', -SVR_search.best_score_)

# 최종 모델로 test 데이터 예측 후 제출 파일 생성
test_pred = SVR_search.predict(test_X)
test_pred = pd.DataFrame(test_pred, columns = ['pred'])

test_pred.to_csv('submission.csv', index = False)
