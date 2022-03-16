from sklearn import svm
import numpy as np

train_path = "./Encoding_result/train_ptm.txt"
#train_path = "/content/drive/MyDrive/PTM_Project/iLearn_AbirModifiedFinal/Encoding_result/train_ptm.txt"
test_path = "./Encoding_result/test_ptm.txt"
#test_path = "/content/drive/MyDrive/PTM_Project/iLearn_AbirModifiedFinal/Encoding_result/test_ptm.txt"

# read and convert train path data to numpy array
train_data = np.loadtxt(train_path, delimiter=',')
#print (train_data[:])
#print(train_data.shape)

# divided train_data two featues and label first index is label
train_label = train_data[:, 0]
train_features = train_data[:, 1:]

test_data = np.loadtxt(test_path, delimiter=',')
#print (test_data[2:4])

# divided test_data two featues and label first index is label
test_label = test_data[:, 0]
test_features = test_data[:, 1:]


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(train_features,train_label)

y_pred=clf.predict(test_features[:])


from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_label, y_pred))