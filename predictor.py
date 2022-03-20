import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

train = "examples/DNA_code_training.txt" #Training dataset
indep = "examples/DNA_code_testing.txt" #Independent dataset
format = "svm" # choices=['tsv', 'svm', 'csv', 'weka']
n_trees = 100 #the number of trees in the forest (default 100)
#fold = 5 #n-fold cross validation mode (default 5-fold cross-validation, 1 means jack-knife cross-validation)
out = 'RF' #set prefix for output score file

X, y, independent = 0, 0, np.array([])
X, y = read_code_ml.read_code(args.train, format='%s' % format)
if indep:
    ind_X, ind_y = read_code_ml.read_code(indep, format='%s' % format)
    independent = np.zeros((ind_X.shape[0], ind_X.shape[1] + 1))
    independent[:, 0], independent[:, 1:] = ind_y, ind_X

para_info, cv_res, ind_res = RF_Classifier.RF_Classifier(X, y, indep=independent, fold=fold, n_trees=n_trees, out=out)
classes = sorted(list(set(y)))
if len(classes) == 2:
    save_file.save_CV_result_binary(cv_res, '%s_CV.txt' % out, para_info)
    mean_auc = draw_plot.plot_roc_cv(cv_res, '%s_ROC_CV.png' % out, label_column=0, score_column=2)
    mean_auprc = draw_plot.plot_prc_CV(cv_res, '%s_PRC_CV.png' % out, label_column=0, score_column=2)
    cv_metrics = calculate_prediction_metrics.calculate_metrics_cv(cv_res, label_column=0, score_column=2,)
    save_file.save_prediction_metrics_cv(cv_metrics, '%s_metrics_CV.txt' % out)
    if indep:
        save_file.save_IND_result_binary(ind_res,'%s_IND.txt' % out, para_info)
        ind_auc = draw_plot.plot_roc_ind(ind_res, '%s_ROC_IND.png' % out, label_column=0, score_column=2)
        ind_auprc = draw_plot.plot_prc_ind(ind_res, '%s_PRC_IND.png' % out, label_column=0, score_column=2)
        ind_metrics = calculate_prediction_metrics.calculate_metrics(ind_res[:, 0], ind_res[:, 2])
        save_file.save_prediction_metrics_ind(ind_metrics, '%s_metrics_IND.txt' % out)

if len(classes) > 2:
    save_file.save_CV_result(cv_res, classes, '%s_CV.txt' % out, para_info)
    cv_metrics = calculate_prediction_metrics.calculate_metrics_cv_muti(cv_res, classes, label_column=0)
    save_file.save_prediction_metrics_cv_muti(cv_metrics, classes, '%s_metrics_CV.txt' % out)

    if indep:
        save_file.save_IND_result(ind_res, classes, '%s_IND.txt' % out, para_info)
        ind_metrics = calculate_prediction_metrics.calculate_metrics_ind_muti(ind_res, classes, label_column=0)
        save_file.save_prediction_metrics_ind_muti(ind_metrics, classes, '%s_metrics_IND.txt' % out)
