import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from pubscripts import read_code_ml, save_file, draw_plot, calculate_prediction_metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

train = "Encoding_result/Train.txt" #Training dataset
indep = "Encoding_result/Test.txt" #Independent dataset

format = "csv" # choices=['tsv', 'svm', 'csv', 'weka']
n_trees = 1000 #the number of trees in the forest (default 100)
#fold = 5 #n-fold cross validation mode (default 5-fold cross-validation, 1 means jack-knife cross-validation)
#out = 'RF' #set prefix for output score file

X, y, independent = 0, 0, np.array([])
X, y = read_code_ml.read_code(train, format='%s' % format)
if indep:
    ind_X, ind_y = read_code_ml.read_code(indep, format='%s' % format)
    independent = np.zeros((ind_X.shape[0], ind_X.shape[1] + 1))
    independent[:, 0], independent[:, 1:] = ind_y, ind_X

## Random Forest

model = RandomForestClassifier(n_estimators=n_trees, bootstrap=False)
rfc = model.fit(X, y)
rf_scores = rfc.predict_proba(X)
rf_scores_ind = rfc.predict_proba(ind_X)
rf_metrix = calculate_prediction_metrics.calculate_metrics(ind_y, rf_scores_ind[:, 1])
print("RF Accuracy for independent dataset:", rf_metrix['Accuracy'])

## SVM
kernel = 'rbf'
degree = 3
gamma = 'auto'
coef0 = 0
C = 1.0
default_params = {'degree': degree, 'gamma': gamma, 'coef0': coef0, 'C': C}
auto = True
batch = None
if auto:
    data = np.zeros((X.shape[0], X.shape[1] + 1))
    data[:, 0] = y
    data[:, 1:] = X
    np.random.shuffle(data)
    X1 = data[:, 1:]
    y1 = data[:, 0]
    parameters = {'kernel': ['linear'], 'C': [1, 15]} if kernel == 'linear' else {'kernel': [kernel],
                                                                                  'C': [1, 15],
                                                                                  'gamma': 2.0 ** np.arange(-10, 4)}
    optimizer = GridSearchCV(svm.SVC(probability=True), parameters)
    optimizer = optimizer.fit(X1[0:math.ceil(batch * X1.shape[0]), ],
                              y1[0:math.ceil(batch * y1.shape[0]), ]) if batch else optimizer.fit(X, y)
    params = optimizer.best_params_
    default_params['C'] = params['C']
    if kernel != 'linear':
        default_params['gamma'] = params['gamma']

print(kernel)
print(default_params['C'])
print(default_params['gamma'])
print(default_params['coef0'])
model = svm.SVC(C=default_params['C'], kernel=kernel, degree=default_params['degree'],
                    gamma=default_params['gamma'], coef0=default_params['coef0'], probability=True,
                    random_state=1)
svc = model.fit(X, y)
#svms.append(svc)
svm_scores = svc.predict_proba(X)
svm_scores_ind = svc.predict_proba(ind_X)
svm_metrix = calculate_prediction_metrics.calculate_metrics(ind_y, svm_scores_ind[:, 1])
print("SVM Accuracy for independent dataset:", svm_metrix['Accuracy'])
#print(svm_scores)

#KNN
n_neighbors = 3
knnc = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
knn_scores = knnc.predict_proba(X)
knn_scores_ind = knnc.predict_proba(ind_X)
knn_metrix = calculate_prediction_metrics.calculate_metrics(ind_y, knn_scores_ind[:, 1])
print("KNN Accuracy for independent dataset:", knn_metrix['Accuracy'])

#print(knn_scores)

#MLP
activation = 'relu'
hidden_layer_size = (32, 32)
lost = 'lbfgs'
epochs = 200
lr = 0.001
model = MLPClassifier(activation=activation, alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999,
                              early_stopping=False, epsilon=1e-08, hidden_layer_sizes=hidden_layer_size,
                              learning_rate='constant', learning_rate_init=lr, max_iter=epochs, momentum=0.9,
                              nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver=lost,
                              tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=False)
mlpc = model.fit(X, y)
mlp_scores = mlpc.predict_proba(X)
mlp_scores_ind = mlpc.predict_proba(ind_X)
mlp_metrix = calculate_prediction_metrics.calculate_metrics(ind_y, mlp_scores_ind[:, 1])
print("MLP Accuracy for independent dataset:", mlp_metrix['Accuracy'])


#print(mlp_scores)


#Combine

combine_scores_X = np.vstack((rf_scores[:, 1], svm_scores[:, 1], knn_scores[:, 1], mlp_scores[:, 1])).T
combine_scores_ind = np.vstack((rf_scores_ind[:, 1], svm_scores_ind[:, 1], knn_scores_ind[:, 1], mlp_scores_ind[:, 1])).T
#print(combine_scores)

model = RandomForestClassifier(n_estimators=n_trees, bootstrap=False)
combine_rfc = model.fit(combine_scores_X, y)
combine_scores_ind = combine_rfc.predict_proba(combine_scores_ind)
#combine_class = combine_rfc.predict(combine_scores_ind)
combine_metrix = calculate_prediction_metrics.calculate_metrics(ind_y, combine_scores_ind[:, 1])
print("Combine RF Accuracy for independent dataset:", combine_metrix['Accuracy'])

