import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import tree

# Generate heatmap to understand the correlation between the variables
#ccdefault = pd.read_csv('D:\Downloads\CPS844\\Raisin_Dataset.csv')
#correlation_matrix = ccdefault.corr().round(2)
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.heatmap(data=correlation_matrix, annot=True)
#plt.show()

#K Nearest Neighbor Classifier
def KNN():
    
    # load the data
    ccdefault = pd.read_csv('Raisin_Dataset.csv')

    # seperate the class from the other attributes
    classData = ccdefault['Class']
    attributeData = ccdefault.drop(['Class'], axis=1)

    # Standardize the data
    attributeData = scale(attributeData)

    # Split the data
    dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData, classData, stratify=classData, random_state=1)

    # KNN classification
    clf = KNeighborsClassifier(n_neighbors = 5)
    clf.fit(dataTrain, classTrain)
    predC = clf.predict(dataTest)

    # Print KNN accuracy
    print("Accuracy:", round(accuracy_score(classTest, predC)*100), "%")
    fig=plot_confusion_matrix(clf, dataTest, classTest,display_labels=["Kecimen","Besni"])
    fig.figure_.suptitle("Confusion Matrix for Raisin Dataset")
    plt.show()

#Artificial Neural Networks
def ANN():

    # load the data
    ccdefault = pd.read_csv('Raisin_Dataset.csv')

    # seperate the class from the other attributes
    classData = ccdefault['Class']
    attributeData = ccdefault.drop(['Class'], axis=1)

    # Standardize the data
    #attributeData = normalize(attributeData)
    attributeData = scale(attributeData)

    # Split the data
    dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData, classData, stratify=classData, random_state = 1)
    
    # ANN classification
    clf = MLPClassifier(max_iter=1000)
    clf.fit(dataTrain, classTrain)
    predC = clf.predict(dataTest)

    # Print ANN accuracy
    print("Accuracy:", round(accuracy_score(classTest, predC)*100), "%")
    fig=plot_confusion_matrix(clf, dataTest, classTest,display_labels=["Kecimen","Besni"])
    fig.figure_.suptitle("Confusion Matrix for Raisin Dataset")
    plt.show()

# Decision Tree classification
def DTC():
    ccdefault = pd.read_csv('Raisin_Dataset.csv')

    # seperate the class from the other attributes
    classData = ccdefault['Class']
    attributeData = ccdefault.drop(['Class'], axis=1)

    # Standardize the data
    #attributeData = normalize(attributeData)
    attributeData = scale(attributeData)
    
    # Split the data
    dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData, classData, test_size=0.3, stratify=classData, random_state=1)

    # Decision Tree classification
    clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
    clf = clf.fit(dataTrain, classTrain)
    predC = clf.predict(dataTest)
    
    print("Accuracy:", round(accuracy_score(classTest, predC)*100), "%")
    tree.plot_tree(clf, fontsize = 7)
    plt.show()
    fig=plot_confusion_matrix(clf, dataTest, classTest,display_labels=["Kecimen","Besni"])
    fig.figure_.suptitle("Confusion Matrix for Raisin Dataset")
    plt.show()
    
# Naive Bayes Classification
def NB():
    # load the data
    ccdefault = pd.read_csv('Raisin_Dataset.csv')

    # seperate the class from the other attributes
    classData = ccdefault['Class']
    attributeData = ccdefault.drop(['Class'], axis=1)

    # Standardize the data
    attributeData = scale(attributeData)
    #attributeData = normalize(attributeData)

    # Split the data
    dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData, classData, stratify=classData, random_state = 1)

    #Naive Bayes Classification
    clf = GaussianNB()
    clf.fit(dataTrain, classTrain)
    predC = clf.predict(dataTest)

    # Print NB accuracy
    print("Accuracy:", round(accuracy_score(classTest, predC)*100), "%")
    fig=plot_confusion_matrix(clf, dataTest, classTest,display_labels=["Kecimen","Besni"])
    fig.figure_.suptitle("Confusion Matrix for Raisin Dataset")
    plt.show()

#Logistic Regression Classification
def LG():
    # load the data
    ccdefault = pd.read_csv('Raisin_Dataset.csv')

    # seperate the class from the other attributes
    classData = ccdefault['Class']
    attributeData = ccdefault.drop(['Class'], axis=1)

    # Standardize the data
    attributeData = scale(attributeData)
    #attributeData = normalize(attributeData)

    # Split the data
    dataTrain, dataTest, classTrain, classTest = train_test_split(attributeData, classData, stratify=classData, random_state=1)

    #Logistic Regression Classification
    clf = LogisticRegression()
    clf.fit(dataTrain, classTrain)
    predC = clf.predict(dataTest)

    #Logistic Regression
    print("Accuracy:", round(accuracy_score(classTest, predC)*100), "%")
    fig=plot_confusion_matrix(clf, dataTest, classTest,display_labels=["Kecimen","Besni"])
    fig.figure_.suptitle("Confusion Matrix for Raisin Dataset")
    plt.show()
