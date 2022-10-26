import pandas as pd
from sklearn.naive_bayes import GaussianNB
from apyori import apriori
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

#Part 1: Naïve Bayes Classifier
#1. (0 points) Load the dataset weather.csv
data = pd.read_csv("weather.csv")
#2. (15 points) As you may have noticed, the implementations of many classifiers do not
#support categorical data. Convert any categorical variable into dummy variables (a dummy
#variable takes only the value 0 or 1 to indicate the absence or presence of some attribute
#values). Hint: research about pandas’ function get_dummies, and ensure the dummy
#variables are of type ‘float’.
data = pd.get_dummies(data).astype('float')
#data = pd.get_dummies(data[['outlook', 'temperature', 'humidity', 'windy', 'play']]).astype('float')


#3. (5 points) At this point, you should notice that the target attribute is now split into 2 target
#attributes. Drop the target attribute 'play_no'
data = data.drop(['play_no'],axis=1)

#4. (5 points) Continue with the preprocessing: separate the features attributes from the target attribute.
classData = data['play_yes']
attributeData = data.drop(['play_yes'], axis=1)

#5. (5 points) Use the sklearn library to construct a Gaussian Naive Bayes classifier, then train this classifier.
nb = GaussianNB()
nb = nb.fit(attributeData, classData)

#6. (15 points) Using the trained classifier, determine if the class label of ‘play_yes’ is more
#likely to be ‘0’ or ‘1’, for a “new day”, where the new day has the following attributes:
#Outlook = sunny, Temperature = 66, Humidity = 90, Windy = true.

data = pd.read_csv("weather.csv")
testData = [['sunny', 66, 90, True]]
testData = pd.DataFrame(testData, columns = ['outlook', 'temperature', 'humidity', 'windy'])
testData = pd.concat([data, testData], ignore_index=True)

testData = pd.get_dummies(testData).astype('float')
testData = testData.drop(['play_no'],axis=1)

testC = testData['play_yes']
testD = testData.drop(['play_yes'],axis=1)
proba = nb.predict_proba(testD)

print("Part 1 ------------------------\n")

if proba[-1][0] > proba[-1][1]:
    print("play_yes is more likely to be 0")
else:
    print("play_yes is more likely to be 1")

print("Likelihood of play = yes: ", (proba[-1][1]))
print("Likelihood of play = no: ", (proba[-1][0]))

#Part 2: Association Analysis
#8. (0 points) Reload the dataset weather.csv if you have modified it.
data = pd.read_csv("weather.csv")

#9. (15 points) Discretize the continuous data
data['temperature'] = pd.cut(x=data['temperature'], bins=3, labels=['cool','mild','hot'])
data['humidity'] = pd.cut(x=data['humidity'], bins=2, labels=['normal','high'])

#10. (10 points) Because of the implementation of the apyori library, you will also need to
#convert the boolean values from the attribute 'windy' to string. Hint: consider calling the
#method ‘map’ on the column ‘windy’ of your dataframe, in order to ‘map’ the boolean
#values to any string values
data['windy'] = data['windy'].map(str)

#11. (5 points) As seen during the lecture, the apyori library requires inputs in the form of a list
#of lists. Convert the whole dataset as a big list, where each ‘record’ is an inner list within
#the big list (you can re-use the code posted for the demo on D2L).
weather = []
for i in range(data.shape[0]):
    weather.append(data.iloc[i].dropna().tolist())

#12. (5 points) You are ready to call the apriori function from the apyori module (you can reuse the code posted for the demo on D2L).
#You can start with a minimum support threshold
#of 0.28 and a minimum confidence threshold of 0.5.
rules = apriori(weather, min_support = 0.28, min_confidence = 0.5)

print("\nPart 2 ------------------------\n")

print("min_support = 0.28, min_confidence = 0.5")
for rule in rules:
    print(list(rule.ordered_statistics[0].items_base), '-->', list(rule.ordered_statistics[0].items_add),
        'Support:',rule.support, 'Confidence:', rule.ordered_statistics[0].confidence )

rules = apriori(weather, min_support = 0.18, min_confidence = 0.3)
print("\nmin_support = 0.18, min_confidence = 0.3")
for rule in rules:
    print(list(rule.ordered_statistics[0].items_base), '-->', list(rule.ordered_statistics[0].items_add),
        'Support:',rule.support, 'Confidence:', rule.ordered_statistics[0].confidence )

rules = apriori(weather, min_support = 0.4, min_confidence = 0.5)
print("\nmin_support = 0.4, min_confidence = 0.5")
for rule in rules:
    print(list(rule.ordered_statistics[0].items_base), '-->', list(rule.ordered_statistics[0].items_add),
        'Support:',rule.support, 'Confidence:', rule.ordered_statistics[0].confidence )
