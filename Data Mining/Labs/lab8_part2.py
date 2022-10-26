# -*- coding: utf-8 -*-
"""
Part 2

Density-Based analysis
Identify high-density clusters separated by regions of low density. 
In the popular DBScan, data points are classified into 3 types:
     core points, border points, and noise points
Classification is applied as a function of two parameters: 
     the radius of the neighborhood size (eps) and the minimum number of points in the neighborhood (minpts).
"""
    
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# (0 point) Import the chameleon.data data
data = pd.read_csv('chameleon.data', delimiter=' ', names=['x','y'])
# Check the data distribution
data.plot.scatter(x='x',y='y')
plt.show()

# (15 points) Apply DBScan: eps set to 15.5 and minpts set to 5. 
DBScanAnalysis = DBSCAN(eps=15.5, min_samples=5).fit(data)
# Concatenate data with cluster labels:
# 1. Convert labels as a pandas dataframe
clustersLabels = pd.DataFrame(DBScanAnalysis.labels_,columns=['Cluster ID'])
# 2. (15 points) Concatenate the dataframes 'data' and 'clustersLabels' (hint: use 'axis = 1' for concatenating along the column axis)
result = pd.concat((data, clustersLabels), axis=1)

# (10 points) Create a scatter plot of the data: 
# each point with coordinates x and y is represented as a dot; 
# use the value in 'Cluster ID' to color the point
# Hint: the command is very similar to the one on line 17
result.plot.scatter(x='x',y='y', colormap='jet', c=result['Cluster ID'])
plt.show()
# (10 points) How many clusters were found? Fill out the blank to tell, and don't include the noise points in the count.
# There are 9 clusters, not including the noise 
