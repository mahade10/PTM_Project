import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from pubscripts import read_code_ml, save_file, draw_plot, calculate_prediction_metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

train = "Encoding_result/Train.txt" #Training dataset
indep = "Encoding_result/Test.txt" #Independent dataset

format = "csv" # choices=['tsv', 'svm', 'csv', 'weka']
n_trees = 1000 #the number of trees in the forest (default 100)
fold = 5 #n-fold cross validation mode (default 5-fold cross-validation, 1 means jack-knife cross-validation)
out = 'RF' #set prefix for output score file

X, y, independent = 0, 0, np.array([])
X, y = read_code_ml.read_code(train, format='%s' % format)
if indep:
    ind_X, ind_y = read_code_ml.read_code(indep, format='%s' % format)
    independent = np.zeros((ind_X.shape[0], ind_X.shape[1] + 1))
    independent[:, 0], independent[:, 1:] = ind_y, ind_X

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
