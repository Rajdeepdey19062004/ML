import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings("ignore")

# MODULE CALL
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# RAW DATA LOAD
data = pd.read_csv(r"cancer_data.csv")
data_final = data[["diagnosis", "radius_mean", "texture_mean"]]

# CONVERT DATA TO LIST
x = data_final.iloc[:, 1:].values
y = data_final.iloc[:, 0].values

# TRAIN MODEL
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ===================== KNN =====================
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
knn = accuracy_score(y_test, y_pred)
print("KNN ACCURACY:", knn * 100, "%")

plt.subplot(4, 3, 1)
plt.title("KNN MODEL")
cmknn = confusion_matrix(y_test, y_pred)
sns.heatmap(cmknn, annot=True)
print("KNN CLASSIFICATION REPORT : ")
print(classification_report(y_test, y_pred))

# ===================== LOGISTIC REGRESSION =====================
logm = LogisticRegression()
logm.fit(x_train, y_train)

logm_pred = logm.predict(x_test)
la = accuracy_score(y_test, logm_pred)
print("LOGISTIC ACCURACY : ", la * 100, "%")

plt.subplot(4, 3, 2)
plt.title("LOGISTIC MODEL")
cm_log = confusion_matrix(y_test, logm_pred)
sns.heatmap(cm_log, annot=True)
print("LOGISTIC CLASSIFICATION REPORT : ")
print(classification_report(y_test, logm_pred))

# ===================== NAIVE BAYES =====================
nb = GaussianNB()
nb.fit(x_train, y_train)

np_pred = nb.predict(x_test)
acnb = accuracy_score(y_test, np_pred)
print("NAIVE BAYES ACCURACY : ", acnb * 100, "%")

plt.subplot(4, 3, 3)
plt.title("NAIVE BAYES MODEL")
cmnb = confusion_matrix(y_test, np_pred)
sns.heatmap(cmnb, annot=True)
print("NAIVE BAYES CLASSIFICATION REPORT : ")
print(classification_report(y_test, np_pred))

# ===================== SVM =====================
svm = SVC(kernel="linear")
svm.fit(x_train, y_train)

svm_pred = svm.predict(x_test)
svmac = accuracy_score(y_test, svm_pred)
print("SVM ACCURACY : ", svmac * 100, "%")

plt.subplot(4, 3, 7)
plt.title("SVM MODEL")
cmsvm = confusion_matrix(y_test, svm_pred)
sns.heatmap(cmsvm, annot=True)
print("SVM CLASSIFICATION REPORT : ")
print(classification_report(y_test, svm_pred))

# ===================== DECISION TREE =====================
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

dt_pred = dt.predict(x_test)
dtac = accuracy_score(y_test, dt_pred)
print("DECISION TREE ACCURACY : ", dtac * 100, "%")

plt.subplot(4, 3, 8)
plt.title("DECISION TREE")
cmdt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cmdt, annot=True)
print("DECISION TREE CLASSIFICATION REPORT : ")
print(classification_report(y_test, dt_pred))

# ===================== RANDOM FOREST =====================
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)
acrf = accuracy_score(y_test, rf_pred)
print("RANDOM FOREST : ", acrf * 100, "%")

plt.subplot(4, 3, 9)
plt.title("RANDOM FOREST")
cmrf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cmrf, annot=True)
print("RANDOM FOREST CLASSIFICATION REPORT : ")
print(classification_report(y_test, rf_pred))


# SHOW ALL GRAPHS
plt.show()
