import pandas as pd
import numpy as np

# 학습용 데이터와 평가용 데이터를 불러온다.
train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_test.csv')

# grade는 예측해야 하는 정답값이므로 입력 변수에서 분리한다.
train_X = train.drop(['grade'],axis=1)
train_y = train['grade']

test_X = test.drop(['grade'], axis=1)
test_y = test['grade']

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV

# 숫자형 컬럼과 문자형 컬럼을 나눠서 서로 다른 전처리를 적용한다.
num_columns = train_X.select_dtypes('number').columns.tolist()
cat_columns = train_X.select_dtypes('object').columns.tolist()

# 문자형 변수는 원핫인코딩한다. 테스트 데이터에 새로운 범주가 나와도 오류가 나지 않게 한다.
cat_preprocess = make_pipeline(
    OneHotEncoder(handle_unknown='ignore', sparse_output= False)
)

# 숫자형 변수는 결측치를 평균으로 채우고 스케일을 맞춘다.
num_preprocess = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

# ColumnTransformer로 숫자형/문자형 전처리를 한 번에 묶는다.
preprocess = ColumnTransformer(
    [
        ('num', num_preprocess, num_columns),
        ('cat', cat_preprocess, cat_columns)
    ]
)

from sklearn.neighbors import KNeighborsRegressor

# KNN 회귀 모델: 전처리 후 가까운 이웃들의 값으로 grade를 예측한다.
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('regressor', KNeighborsRegressor())
    ]
)

print(KNeighborsRegressor().get_params())

# 파이프라인 안 모델의 옵션은 단계명__파라미터명 형태로 지정한다.
knn_param = {'regressor__n_neighbors' : np.arange(5,10,1)}

# GridSearchCV로 여러 n_neighbors 값을 비교한다.
knn_search = GridSearchCV(
    estimator= full_pipe,
    param_grid= knn_param,
    cv = 3,
    scoring= 'neg_mean_squared_error'
)
knn_search.fit(train_X,train_y)

pd.DataFrame(knn_search.cv_results_)
print('Best 파라미터 조합: ', knn_search.best_params_)
# scoring이 neg_mean_squared_error라서 -를 붙여 양수 MSE로 바꾼다.
print('교차검증 MSE: ', -knn_search.best_score_)

from sklearn.metrics import mean_squared_error
knn_pred = knn_search.predict(test_X)
print('테스트 MSE: ', mean_squared_error(test_y, knn_pred))

from sklearn.tree import DecisionTreeRegressor

# 의사결정나무 회귀 모델: 가지치기 강도(ccp_alpha)를 바꿔가며 비교한다.
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('regressor', DecisionTreeRegressor())
    ]
)

print(DecisionTreeRegressor().get_params())

decisiontree_param = {'regressor__ccp_alpha' : np.arange(0.01,0.3,0.05)}

# 단일 트리 모델은 과적합되기 쉬우므로 가지치기 파라미터를 튜닝한다.
decisiontree_search = GridSearchCV(
    estimator= full_pipe,
    param_grid = decisiontree_param,
    cv = 5,
    scoring = 'neg_mean_squared_error'
)
decisiontree_search.fit(train_X,train_y)

print('BEST 파라미터 조합: ', decisiontree_search.best_params_)
print('교차검증 MSE: ', -decisiontree_search.best_score_)

from sklearn.metrics import mean_squared_error
dt_pred = decisiontree_search.predict(test_X)
print('테스트 MSE: ', mean_squared_error(test_y, dt_pred))

from sklearn.ensemble import BaggingRegressor

# 배깅 회귀 모델: 여러 모델을 학습시켜 평균을 내는 앙상블 방식이다.
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('regressor', BaggingRegressor())
    ]
)
print(BaggingRegressor().get_params())

# n_estimators는 앙상블에 사용할 기본 모델 개수다.
Bagging_param = {'regressor__n_estimators' : np.arange(10,100,20),
                 'regressor__random_state' : [0]}
Bagging_search = GridSearchCV(
    estimator = full_pipe,
    param_grid = Bagging_param,
    cv = 5,
    scoring = 'neg_mean_squared_error'
)
Bagging_search.fit(train_X,train_y)

print('Best 파라미터: ', Bagging_search.best_params_)
print('교차검증 MSE score: ', -Bagging_search.best_score_)

from sklearn.metrics import mean_squared_error
bag_pred = Bagging_search.predict(test_X)
print('테스트 MSE: ', mean_squared_error(test_y, bag_pred))

from sklearn.ensemble import RandomForestRegressor

# 랜덤포레스트 회귀 모델: 여러 의사결정나무를 만들고 평균으로 예측한다.
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('regressor', RandomForestRegressor())
    ]
)

print(RandomForestRegressor().get_params())

# max_features는 각 트리가 분기할 때 사용할 후보 변수 개수를 조절한다.
RandomForest_param = {'regressor__n_estimators' : np.arange(100,500,100),
                      'regressor__max_features' : ['sqrt'],
                      'regressor__random_state' : [0]}

RandomForest_search = GridSearchCV(
    estimator = full_pipe,
    param_grid = RandomForest_param,
    cv = 5,
    scoring = 'neg_mean_squared_error'
)
RandomForest_search.fit(train_X,train_y)

print('BEST 파라미터 조합: ', RandomForest_search.best_params_)
print('교차검증 MSE: ', -RandomForest_search.best_score_)

from sklearn.metrics import mean_squared_error
rf_pred = RandomForest_search.predict(test_X)
print('테스트 MSE: ', mean_squared_error(test_y, rf_pred))

from sklearn.ensemble import GradientBoostingRegressor

# 그래디언트 부스팅 회귀 모델: 이전 모델의 오차를 줄이는 방향으로 모델을 순차적으로 쌓는다.
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('regressor', GradientBoostingRegressor())
    ]
)

# learning_rate는 각 단계에서 새 모델을 얼마나 강하게 반영할지 정한다.
GradientBoosting_param = {'regressor__learning_rate' : np.arange(0.1,0.3,0.05),
                          'regressor__random_state' : [0]}
GradientBoosting_search = GridSearchCV(
    estimator = full_pipe,
    param_grid = GradientBoosting_param,
    cv = 5,
    scoring = 'neg_mean_squared_error'
)
GradientBoosting_search.fit(train_X,train_y)

print('Best 파라미터 조합: ', GradientBoosting_search.best_params_)
print('교차검증 MSE: ', -GradientBoosting_search.best_score_)

from sklearn.metrics import mean_squared_error
gb_pred = GradientBoosting_search.predict(test_X)
print('테스트 MSE: ', mean_squared_error(test_y, gb_pred))

from sklearn.svm import SVR

# SVR 회귀 모델: 전처리된 데이터를 이용해 서포트 벡터 방식으로 예측한다.
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('regressor', SVR())
    ]
)

# C는 오차를 얼마나 허용할지 조절하는 SVR의 주요 파라미터다.
SVR_param = {'regressor__C' : np.arange(1,100,20)}
SVR_search = GridSearchCV(
    estimator = full_pipe,
    param_grid = SVR_param,
    cv = 5,
    scoring = 'neg_mean_squared_error'
)
SVR_search.fit(train_X, train_y)

print('Best 파라미터 조합: ', SVR_search.best_params_)
print('교차검증 MSE: ', -SVR_search.best_score_)

from sklearn.metrics import mean_squared_error
SVR_pred = SVR_search.predict(test_X)
print('테스트 MSE: ', mean_squared_error(test_y,SVR_pred))

# 지금부터는 모범 답안 작성 예시
# 위쪽은 Pipeline으로 전처리와 모델을 묶은 방식이고, 아래쪽은 전처리를 직접 수행하는 방식이다.
import pandas as pd
import numpy as np

# 시험 답안처럼 데이터를 다시 불러와 처음부터 절차를 구성한다.
train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_test.csv')

# 데이터 구조, 결측치, 컬럼 타입을 확인한다.
print(train.info())
print(test.info())

# train 데이터에서 정답 컬럼 grade를 분리한다.
train_X = train.drop(['grade'], axis=1)
train_y = train['grade']
from sklearn.model_selection import train_test_split

# 검증 성능을 확인하기 위해 학습용 데이터 일부를 valid 데이터로 나눈다.
train_X, valid_X, train_y, valid_y = train_test_split(
    train_X,
    train_y,
    test_size=0.3,
    random_state= 1
)
print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)

# 수동 전처리를 위해 문자형 컬럼과 숫자형 컬럼을 다시 구분한다.
cat_columns = train_X.select_dtypes('object').columns
num_columns = train_X.select_dtypes('number').columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# 숫자형 결측치 처리를 위한 평균 대체기와 문자형 인코더를 만든다.
imputer = SimpleImputer(strategy = 'mean')
onehotencoder = OneHotEncoder(handle_unknown='ignore', sparse_output= False)

# 숫자형 컬럼은 train에 fit하고, valid/test에는 transform만 적용한다.
train_X_numeric_imputed = imputer.fit_transform(train_X[num_columns])
valid_X_numeric_imputed = imputer.transform(valid_X[num_columns])
test_X_numeric_imputed = imputer.transform(test_X[num_columns])

# 문자형 컬럼도 train에 fit하고, valid/test에는 transform만 적용한다.
train_X_categorical_encoded = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_categorical_encoded = onehotencoder.transform(valid_X[cat_columns])
test_X_categorical_encoded = onehotencoder.transform(test_X[cat_columns])

# 숫자형 처리 결과와 문자형 인코딩 결과를 옆으로 붙여 최종 입력 데이터를 만든다.
train_X_preprocessed = np.concatenate([train_X_numeric_imputed, train_X_categorical_encoded], axis = 1)
valid_X_preprocessed = np.concatenate([valid_X_numeric_imputed, valid_X_categorical_encoded], axis = 1)
test_X_preprocessed = np.concatenate([test_X_numeric_imputed, test_X_categorical_encoded], axis = 1)

from sklearn.ensemble import RandomForestRegressor

# 기본 랜덤포레스트 모델을 학습하고 검증 데이터로 성능을 확인한다.
rf = RandomForestRegressor(random_state = 1)
rf.fit(train_X_preprocessed,train_y)

from sklearn.metrics import mean_squared_error
pred_val = rf.predict(valid_X_preprocessed)
print('vaild RMSE: ', mean_squared_error(valid_y, pred_val, squared= False))

# 테스트 데이터 예측값을 제출 형식으로 저장한다.
test_pred = rf.predict(test_X_preprocessed)
test_pred = pd.DataFrame(test_pred, columns = ['pred'])
test_pred.to_csv('result.csv', index = False)

# 하이퍼파라미터 튜닝을 위해 train과 valid를 다시 합쳐 전체 학습 데이터로 사용한다.
train_X_full = np.concatenate([train_X_preprocessed, valid_X_preprocessed], axis = 0)
train_y_full = np.concatenate([train_y, valid_y], axis = 0)

# 랜덤포레스트의 깊이와 분할 기준을 바꿔가며 비교한다.
param_grid = {'max_depth' : [10,20,30,],
              'min_samples_split' : [2,5,10]}

rf = RandomForestRegressor(random_state = 1)

# 수동 전처리가 끝난 배열을 사용하므로 여기서는 Pipeline 없이 GridSearchCV를 적용한다.
rf_search = GridSearchCV(
    estimator = rf,
    param_grid = param_grid,
    cv = 5,
    scoring = 'neg_mean_squared_error'
)

rf_search.fit(train_X_full, train_y_full)
print('교차검증 MSE: ', -rf_search.best_score_)

# 튜닝된 모델로 테스트 데이터를 예측하고 두 번째 제출 파일로 저장한다.
rf_pred = rf_search.predict(test_X_preprocessed)
rf_pred = pd.DataFrame(rf_pred, columns = ['pred'])
rf_pred.to_csv('result2.csv', index = False)
