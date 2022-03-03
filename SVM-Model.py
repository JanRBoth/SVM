
import pandas as pd
import sklearn
from sklearn import svm
from sklearn import metrics
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from tqdm import tqdm
#Prozess Zeit
start_time = datetime.now()

print("# Dataset Erstellen")
# Dataset
path_read = "../../../../1. Data/1. Agri/Feature_Farbe.csv"
data = pd.read_csv(path_read, sep='/') #Einlesesen der CSV Datei
x = data[data.columns[1:769]]
y = data.Label
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# SVM Model / Train
print("# Train SVM Model Linear")
clf_lin = svm.SVC(kernel="linear")
clf_lin.fit(x_train, y_train)

print("# Train SVM Model Non Linear")
# Non-linear SVM Model / Train
clf_nu = svm.NuSVC(gamma="auto")
clf_nu.fit(x_train, y_train)

print("# Predict")
# Predict
y_pred_lin = clf_lin.predict(x_test)
y_pred_nu = clf_nu.predict(x_test)

print("# Metrics")
# Metrics
acc_lin = metrics.accuracy_score(y_test, y_pred_lin)
prec_lin = precision_score(y_test, y_pred_lin, average='micro')
confm_lin = confusion_matrix(y_test, y_pred_lin, labels=[0, 1])
print("# SVM - Linear #")
print("Accuracy: ", acc_lin)
print("Precision: ", prec_lin)
print("Confusion Matrix")
print(confm_lin)

print("#############################################")
acc_nu = metrics.accuracy_score(y_test, y_pred_nu)
prec_nu = precision_score(y_test, y_pred_nu, average='micro')
confm_nu = confusion_matrix(y_test, y_pred_nu, labels=[0, 1])
print("# SVM - Non Linear #")
print("Accuracy: ", acc_nu)
print("Precision: ", prec_nu)
print("Confusion Matrix")
print(confm_nu)