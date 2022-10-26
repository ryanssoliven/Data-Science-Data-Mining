# Imports
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt

# 1. (0 points) Load the dataset
moviesDataset = pd.read_csv(r'DataLab7.csv')

# 2. (5 points) To perform a k-means analysis on the dataset, extract only the numerical attributes: remove the "user" attribute 
data = moviesDataset.drop('user',axis=1)

## Suppose you want to determine the number of clusters k in the initial data 'data' ##
# 3. (5 points) Create an empty list to store the SSE of each value of k (so that, eventually, we will be able to compute the optimum number of clusters k)
SSE = []

# 4. (30 points) Apply k-means with a varying number of clusters k and compute the corresponding sum of squared errors (SSE) 
# Hint1: use a loop to try different values of k. Think about the reasonable range of values k can take (for example, 0 is probably not a good idea).
# Hint2: research about cluster.KMeans and more specifically 'inertia_'
# Hint3: If you get an AttributeError: 'NoneType' object has no attribute 'split', consider downgrading numpy to 1.21.4 this way: pip install --upgrade numpy==1.21.4
k = range(1,6)
for clusterNumber in k:
    k_means_analysis = cluster.KMeans(n_clusters=clusterNumber)
    k_means_analysis.fit(data)
    SSE.append(k_means_analysis.inertia_)

#  5. (20 points) Plot to find the SSE vs the Number of Cluster to visually find the "elbow" that estimates the number of clusters
plt.plot(k, SSE)
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')


# 6. (10 points) Look at the plot and determine the number of clusters k (value of the "elbow" as explained in lecture)
k = 2

# 7. (30 points) Using the optimized value for k, apply k-means on the data to partition the data, then store the labels in a variable named 'labels'
# Hint1: research about cluster.KMeans and more specifically 'labels_'
k_means = cluster.KMeans(n_clusters=k, random_state=1)
k_means.fit(data) 
labels = k_means.labels_

# 8. Display the assignments of each users to a cluster 
clusters = pd.DataFrame(labels, index=moviesDataset.user, columns=['Cluster ID'])
