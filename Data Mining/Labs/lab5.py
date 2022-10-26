import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# 2) Read the dataset located here 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)

# 3)
# Assign new headers to the DataFrame
data.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses','Class']

# Drop the 'Sample code number' attribute
data = data.drop(['Sample code number'],axis=1)

# Convert the '?' to NaN
data = data.replace('?',np.NaN)

# Discard the data points that contain missing values
data = data.dropna()

# Drop row duplicates
data = data.drop_duplicates()

# 4)
# Separate the features from the target class
classData = data['Class']
attributeData = data.drop(['Class'], axis=1)

# Standardize the features
attributeData = preprocessing.scale(attributeData)

#Modify the target values: from the description, the malignant class
#labels are indicated with the value 4. The ‘benign’ labels are indicated with the value ‘2’.
#Replace (or ‘map’) the values such as the label ‘4’ becomes the integer ‘1’, (malignant class), 
#and the label ‘2’ becomes the integer ‘0’ (benign class).
classData = classData.replace(4, 1)
classData = classData.replace(2, 0)

# 5) Use the sklearn library to construct a NearestNeighbors classifier. Keep the default value for the number of neighbors 
dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData, classData, test_size = 0.4, random_state=1)
clf = KNeighborsClassifier(n_neighbors = 5)
clf.fit(dataTrain, classTrain)
predC = clf.predict(dataTest)

# 6)
#Compute and print out the averages of the accuracies, 10-fold cross validation
scores = cross_val_score(clf, attributeData, classData, cv=10)
print("Average of accuracy: ", scores.mean())

#Compute and print out the averages of the f1-scores, 10-fold cross validation
scores = cross_val_score(clf, attributeData, classData, cv=10, scoring='f1')
print("Average of fl-scores: ", scores.mean())

#Compute and print out the averages of the precisions, 10-fold cross validation
scores = cross_val_score(clf, attributeData, classData, cv=10, scoring='precision')
print("Average of precisions: ", scores.mean())

#Compute and print out the averages of the recall measurements, 10-fold cross validation
scores = cross_val_score(clf, attributeData, classData, cv=10, scoring='recall')
print("Average of recall measurements: ", scores.mean())

# 7) Confusion matrix
fig=plot_confusion_matrix(clf, dataTest, classTest,display_labels=["Benign","Malignant"])
fig.figure_.suptitle("Confusion Matrix for Dataset")
plt.show()
