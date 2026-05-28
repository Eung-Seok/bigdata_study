import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target

df = pd.DataFrame(X, columns = data.feature_names)
df['target'] = y

print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = 'target'),
                                                    df['target'],
                                                    test_size=0.3,
                                                    random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 10000, random_state = 0)
model.fit(X_train, y_train)

y_prob_org = model.predict_proba(X_test)
print(pd.DataFrame(y_prob_org[:4].round(3)))

y_pred = model.predict(X_test)
print(pd.DataFrame(y_pred, columns = ['pred']).head(4))

y_pred_ths = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print('값이 같은지 확인: ', np.array_equal(y_pred_ths, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# cm = confusion_matrix(y_test, y_pred)
# isp = ConfusionMatrixDisplay(confusion_matrix=cm)
# isp.plot(cmap=plt.cm.Blues)
# plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

print(accuracy)
print(precision)
print(recall)
print(f1)

print('------------------------------------------')
precision2 = precision_score(y_test,y_pred, pos_label = 1)
recall2 = recall_score(y_test,y_pred, pos_label=1)
f12 = f1_score(y_test,y_pred,pos_label = 1)

print(precision2)
print(recall2)
print(f12)
print('------------------------------------------')

precision3 = precision_score(y_test,y_pred,pos_label=1,average='macro')
recall3 = recall_score(y_test,y_pred,pos_label=1,average='macro')
f13 = f1_score(y_test,y_pred,pos_label=1,average='macro')

print(precision3)
print(recall3)
print(f13)
print('------------------------------------------')

y_pred_ths1 = (model.predict_proba(X_test)[:,1] >= 0.1).astype(int)
y_pred_ths2 = (model.predict_proba(X_test)[:,1] >= 0.9).astype(int)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred_ths1)
cm2 = confusion_matrix(y_test,y_pred_ths2)

print('임계값 0.1일 때 : \n', cm1)
print('임계값 0.9일 때: \n', cm2)

from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_test)[:,1]
auc_score = roc_auc_score(y_test, y_prob)

print('AUC score: %f' % auc_score)

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

print(conf_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred,average='macro')
recall_micro = recall_score(y_test,y_pred, average='micro')
f1_weighted = f1_score(y_test,y_pred, average='weighted')

print(accuracy)
print(precision_macro)
print(recall_micro)
print(f1_weighted)

from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test)
auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
print(f'AUC Score (One-vs-Rest, Macro Average): {auc:.4f}')

train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/s13_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/s13_test.csv')
print(train.head(3))

train_X = train.drop(['diagnosis'], axis=1)
train_y = train['diagnosis']

test_X = test.drop(['diagnosis'], axis=1)
test_y = test['diagnosis']

print(train_y.head(3))

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(train_X, train_y)

y_pred = model.predict(test_X)
from sklearn.metrics import classification_report
print(classification_report(test_y, y_pred))

from sklearn.metrics import f1_score
# f1 = f1_score(test_y, y_pred) 오류나는거
f1 = f1_score(test_y, y_pred, pos_label='A')
print(f'Test set F1 score: {f1:.2f}')

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors' : [3,5,7,9,11]}
grid_search = GridSearchCV(
    model, param_grid, cv = 3, scoring = 'f1'
)

grid_search.fit(train_X, train_y)

print('Best 파라미터 조합: ', grid_search.best_params_)
print('교차검증 F1: ', grid_search.best_score_ )
#긍정 클래스가 A,B중에 어떤건지 안해놔서 nan으로 출력된거임
print(pd.DataFrame(grid_search.cv_results_))

train_y2 = train_y.map({'A': 1, 'B' : 0})
test_y2 = test_y.map({'A': 1, 'B' : 0})

from sklearn.preprocessing import LabelEncoder
labels = ['A','B']

labelencoder = LabelEncoder()
encoded_labels = labelencoder.fit_transform(labels)

print(f'Original label: {labels}')
print(f'Encoded labels: {encoded_labels}')
print(f'Classes: {labelencoder.classes_}')

train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/wisconsin_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/wisconsin_test.csv')

train_X = train.drop(['diagnosis'], axis = 1)
train_y = train['diagnosis']

test_X = test.drop(['diagnosis'], axis = 1)
test_y = test['diagnosis']

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

train_y = labelencoder.fit_transform(train_y)
test_y = labelencoder.transform(test_y)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV

num_columns = train_X.select_dtypes('number').columns.tolist()

num_process = make_pipeline(
    StandardScaler(),
    PCA(n_components = 0.8, svd_solver = 'full')
)

preprocess = ColumnTransformer(
    [
        ('num', num_process, num_columns)
    ]
)

from sklearn.neighbors import KNeighborsClassifier 
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('classifier', KNeighborsClassifier())
    ]
)

knn_params = {'classifier__n_neighbors' : np.arange(5,10,1)}
knn_search = GridSearchCV(estimator= full_pipe,
                          param_grid = knn_params,
                          cv = 3,
                          scoring = 'f1_macro')
knn_search.fit(train_X, train_y)
print('Best 파라미터 조합: ', knn_search.best_params_)
print('교차검증 f1 스코어: ', knn_search.best_score_)

from sklearn.metrics import f1_score
knn_pred = knn_search.predict(test_X)

print('테스트 f1-score: ', f1_score(test_y, knn_pred))

from sklearn.tree import DecisionTreeClassifier
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('classifier', DecisionTreeClassifier())
    ]
)

decisiontree_param = {'classifier__ccp_alpha' : np.arange(0.01,0.3,0.05)}
decisiontree_search = GridSearchCV(
    estimator= full_pipe,
    param_grid = decisiontree_param,
    cv = 5,
    scoring = 'roc_auc'
)
decisiontree_search.fit(train_X, train_y)

print('Best 파라미터 조합: ', decisiontree_search.best_params_)
print('교차검증 AUC: ', decisiontree_search.best_score_)

from sklearn.metrics import roc_auc_score
y_prob = decisiontree_search.predict_proba(test_X)[:,1]
auc_score = roc_auc_score(test_y, y_prob)
print('AUC score: %f' % auc_score)

from sklearn.ensemble import BaggingClassifier

full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('classifier', BaggingClassifier())
    ]
)

Bagging_param = {'classifier__n_estimators' : np.arange(10,100,20)}
Bagging_search = GridSearchCV(
    estimator = full_pipe,
    param_grid = Bagging_param,
    cv = 5,
    scoring = 'f1_macro'
)
Bagging_search.fit(train_X,train_y)

print('Best 파라미터 조합: ', Bagging_search.best_params_)
print('교차검증 f1 score:', Bagging_search.best_score_)

from sklearn.metrics import f1_score
bag_pred = Bagging_search.predict(test_X)
print('테스트 f1 score: ', f1_score(test_y, bag_pred))

from sklearn.ensemble import RandomForestClassifier

full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('classifier', RandomForestClassifier())
    ]
)

RandomForest_param = {'classifier__n_estimators' : np.arange(100,500,100)}
RandomForest_search = GridSearchCV(
    estimator = full_pipe,
    param_grid = RandomForest_param,
    cv = 3,
    scoring = 'accuracy'
)
RandomForest_search.fit(train_X, train_y)

print('Best 파라미터 조합: ', RandomForest_search.best_params_)
print('교차검증 accuracy score: ', RandomForest_search.best_score_)

from sklearn.metrics import accuracy_score
rf_pred = RandomForest_search.predict(test_X)
print('테스트 accuracy score: ', accuracy_score(test_y,rf_pred))

from sklearn.ensemble import GradientBoostingClassifier
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('classifier', GradientBoostingClassifier())
    ]
)
GradientBoosting_param = {'classifier__learning_rate' : np.arange(0.1,0.3,0.05)}
GradientBoosting_search = GridSearchCV(
    estimator = full_pipe,
    param_grid = GradientBoosting_param,
    cv = 5,
    scoring = 'f1_macro'
)
GradientBoosting_search.fit(train_X, train_y)

print('Best 파라미터 조합: ', GradientBoosting_search.best_params_)
print('교차검증 f1 score: ', GradientBoosting_search.best_score_)

from sklearn.metrics import f1_score
gb_pred = GradientBoosting_search.predict(test_X)
print('테스트 f1 score: ', f1_score(test_y, gb_pred))

from sklearn.svm import SVC
full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('classifier', SVC(probability = True))
    ]
)
SVC_param = {'classifier__C' : np.arange(1,100,20)}
SVC_search = GridSearchCV(
    estimator=full_pipe,
    param_grid = SVC_param,
    cv = 3,
    scoring = 'roc_auc'
)
SVC_search.fit(train_X,train_y)

print('Best 파라미터 조합: ', SVC_search.best_params_)
print('교차검증 AUC score: ', SVC_search.best_score_)

from sklearn.metrics import roc_auc_score
SVC_pred = SVC_search.predict(test_X)
print('테스트 AUC score: ', roc_auc_score(test_y,SVC_pred))

#지금부터는 모범 답안

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

X = data.data
y = data.target

df = pd.DataFrame(X, columns = data.feature_names)
df['target'] = y

print(df.head())

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
    df.drop(columns = 'target'),
    df['target'],
    test_size = 0.3,
    random_state=42
)
print(train_X.info())
print(test_X.info())

train_X, valid_X, train_y, valid_y = train_test_split(
    train_X,
    train_y,
    test_size = 0.3,
    random_state = 1
)

cat_columns = train_X.select_dtypes('object').columns
num_columns = train_X.select_dtypes('number').columns

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)

train_X_categorical_encoded = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_categorical_encoded = onehotencoder.transform(valid_X[cat_columns])
test_X_categorical_encoded = onehotencoder.transform(test_X[cat_columns])

train_X_preprocessed = np.concatenate([train_X[num_columns], train_X_categorical_encoded], axis=1)
valid_X_preprocessed = np.concatenate([valid_X[num_columns], valid_X_categorical_encoded], axis=1)
test_X_preprocessed = np.concatenate([test_X[num_columns], test_X_categorical_encoded], axis=1)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 1)
rf.fit(train_X_preprocessed, train_y)

from sklearn.metrics import f1_score
pred_val = rf.predict(valid_X_preprocessed)
print(f1_score(valid_y, pred_val, average='macro'))

test_pred = rf.predict(test_X_preprocessed)
test_pred = pd.DataFrame(test_pred, columns = ['pred'])

test_pred.to_csv('result.csv', index = False)

from sklearn.model_selection import GridSearchCV

train_X_full = np.concatenate([train_X_preprocessed, valid_X_preprocessed], axis = 0)
train_y_full = np.concatenate([train_y, valid_y], axis = 0)

param_grid = {'max_depth' : [10,20,30],
              'min_samples_split' : [2,5,10]}

rf = RandomForestClassifier(random_state = 1)
rf_search = GridSearchCV(
    estimator = rf,
    param_grid = param_grid,
    cv = 3,
    scoring = 'f1_macro'
)

rf_search.fit(train_X_full, train_y_full)
print('교차검증 fi-score: ', rf_search.best_score_)

test_pred2 = rf_search.predict(test_X_preprocessed)
test_pred2 = pd.DataFrame(test_pred2, columns=['pred'])
test_pred2.to_csv('result.csv', index = False)

#아래는 ColumnTransformer와 Pipeline을 사용한 방법
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer

data = load_breast_cancer()

X = data.data
y = data.target

df = pd.DataFrame(X, columns = data.feature_names)
df['target'] = y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns = ['target']),
    df['target'],
    test_size = 0.3,
    random_state = 42
)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

num_columns = train_X.select_dtypes('number').columns.tolist()
cat_columns = train_X.select_dtypes('object').columns.tolist()

num_preprocess = make_pipeline(
    StandardScaler(),
    PCA(n_components = 0.8, svd_solver='full')
)

preprocess = ColumnTransformer(
    [
        ('num', num_preprocess, num_columns)
    ]
)

from sklearn.ensemble import RandomForestClassifier

full_pipe = Pipeline(
    [
        ('preprocess', preprocess),
        ('classifier', RandomForestClassifier())
    ]
)

RandomForest_param = {'classifier__n_estimators' : np.arange(100,500,100)}
RandomForest_search = GridSearchCV(
    estimator= full_pipe,
    param_grid = RandomForest_param,
    cv = 3,
    scoring = 'f1_macro'
)
RandomForest_search.fit(train_X,train_y)

print('Best 파라미터 조합: ', RandomForest_search.best_params_)
print('교차검증 f1-score: ', RandomForest_search.best_score_)
from sklearn.metrics import f1_score
rf_pred = RandomForest_search.predict(test_X)
rf_pred = pd.DataFrame(rf_pred, columns = ['pred'])
rf_pred.to_csv('result.csv', index = False)