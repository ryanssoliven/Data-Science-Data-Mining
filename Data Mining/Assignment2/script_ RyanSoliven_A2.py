import pandas as pd
import numpy as np
from apyori import apriori
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN

#Association Analysis

#Load the Data
data = pd.read_csv('tic-tac-toe.data', delimiter=',', \
                   names=['top-left-square', 'top-middle-square', \
                                                             'top-right-square', 'middle-left-square', \
                          'middle-middle-square', 'middle-right-square', \
                          'bottom-left-square', 'bottom-middle-square', \
                          'bottom-right-square', 'Class'])

# Each record should be a list, and the complete dataset is also a list
tictactoe = []
for i in range(data.shape[0]):
    tictactoe.append(data.iloc[i].tolist())

rule_list = apriori(tictactoe, min_support = 0.4, min_confidence = 0.5)

# we want to print out each rule that was generated, along with its support and confidence
for rule in rule_list:
    print(list(rule.ordered_statistics[0].items_base), '-->', list(rule.ordered_statistics[0].items_add),
        'Support:',rule.support, 'Confidence:', rule.ordered_statistics[0].confidence )

#K-means clustering

data1 = pd.read_csv(r'Acoustic Features.csv')
data = data1.drop('Class',axis=1)

#Create an empty list to store the SSE of each value of k
SSE = []

k = range(1,6)
for clusterNumber in k:
    k_means_analysis = cluster.KMeans(n_clusters=clusterNumber)
    k_means_analysis.fit(data)
    SSE.append(k_means_analysis.inertia_)

#Plot to find the SSE vs the Number of Cluster to visually find the "elbow" that estimates the number of clusters
plt.plot(k, SSE)
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

#apply k-means on the data to partition the data, then store the labels in a variable named 'labels'
k = 2
k_means = cluster.KMeans(n_clusters=k, random_state=1)
k_means.fit(data) 
labels = k_means.labels_

#Display the assignments of each class to a cluster
clusters = pd.DataFrame(labels, index=data1.Class, columns=['Cluster ID'])
clusters=clusters.reset_index()
relax = [0,0]
happy = [0,0]
sad = [0,0]
angry=[0,0]
for index, row in clusters.iterrows():
	if row['Class'] == 'relax':
		if row['Cluster ID'] == 0:
			relax[0] += 1
		else:
			relax[1] += 1
	elif row['Class'] == 'happy':
		if row['Cluster ID'] == 0:
			happy[0] += 1
		else:
			happy[1] += 1
	elif row['Class'] == 'sad':
		if row['Cluster ID'] == 0:
			sad[0] += 1
		else:
			sad[1] += 1
	elif row['Class'] == 'angry':
		if row['Cluster ID'] == 0:
			angry[0] += 1
		else:
			angry[1] += 1
print("\n---------------------------")
print("K-Means Clustering")
print("---------------------------\n")
print(clusters['Cluster ID'].value_counts(),'\n')
print("relax: ", relax)
print("happy: ", happy)
print("sad: ", sad)
print("angry: ", angry)

#Hierarchical Clustering

data = pd.read_csv(r'Acoustic Features.csv')
NumericalAttributes = data.drop(['Class'], axis=1)

#Single link (MIN) analysis + plot associated dendrogram
min_analysis = hierarchy.single(NumericalAttributes)
dn = hierarchy.dendrogram(min_analysis, labels = data['Class'].to_list(), orientation='right')
plt.title("Dendrograms")  
plt.show()

#Complete Link (MAX) analysis + plot associated dendrogram
max_analysis = hierarchy.complete(NumericalAttributes)
dn = hierarchy.dendrogram(max_analysis, labels = data['Class'].to_list(), orientation='right')
plt.show()

#Group Average analysis
average_analysis = hierarchy.average(NumericalAttributes)
dn = hierarchy.dendrogram(average_analysis, labels = data['Class'].to_list(), orientation='right')
plt.show()

print("\n---------------------------")
print("Hierarchical Clustering")
print("---------------------------\n")

#Density-Based Clustering

data = pd.read_csv(r'Acoustic Features.csv')
data = data[['_RMSenergy_Mean', '_Roughness_Mean']]

DBScanAnalysis = DBSCAN(eps=15.5, min_samples=5).fit(data)
clustersLabels = pd.DataFrame(DBScanAnalysis.labels_,columns=['Cluster ID'])
result = pd.concat((data, clustersLabels), axis=1)
result.plot.scatter(x='_RMSenergy_Mean',y='_Roughness_Mean', colormap='jet', c=result['Cluster ID'])
plt.show()

print("\n---------------------------")
print("Density-Based Clustering")
print("---------------------------\n")

