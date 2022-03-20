import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from pubscripts import read_code_ml, save_file, draw_plot, calculate_prediction_metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

train = "Encoding_result/train_ptm.txt" #Training dataset
indep = "Encoding_result/test_ptm.txt" #Independent dataset

format = "csv" # choices=['tsv', 'svm', 'csv', 'weka']
n_trees = 100 #the number of trees in the forest (default 100)
fold = 5 #n-fold cross validation mode (default 5-fold cross-validation, 1 means jack-knife cross-validation)
out = 'RF' #set prefix for output score file

X, y, independent = 0, 0, np.array([])
X, y = read_code_ml.read_code(train, format='%s' % format)
if indep:
    ind_X, ind_y = read_code_ml.read_code(indep, format='%s' % format)
    independent = np.zeros((ind_X.shape[0], ind_X.shape[1] + 1))
    independent[:, 0], independent[:, 1:] = ind_y, ind_X

## Random Forest

print(y[110:120])

model = RandomForestClassifier(n_estimators=n_trees, bootstrap=False)
rfc = model.fit(X, y)
rf_scores = rfc.predict_proba(X)
print(rf_scores[110:120])


