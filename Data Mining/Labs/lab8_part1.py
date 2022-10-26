# -*- coding: utf-8 -*-
"""
Part 1
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy

# (0 point) Import the vertebrate.csv data
data = pd.read_csv('vertebrate.csv')

# (5 points) Pre-process data: create a new variable and bind it with all the numerical attributes (i.e. all except the 'Name' and 'Class')
NumericalAttributes = data.drop(['Name', 'Class'], axis=1)

### (10 points) Single link (MIN) analysis + plot associated dendrogram ###
min_analysis = hierarchy.single(NumericalAttributes)

# (5 points) Plot the associated dendrogram. 
# Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
dn = hierarchy.dendrogram(min_analysis, labels = data['Name'].to_list(), orientation='right')
plt.show()

### (10 points) Complete Link (MAX) analysis + plot associated dendrogram ###
max_analysis = hierarchy.complete(NumericalAttributes)

# (5 points) Plot the associated dendrogram. 
# Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
dn = hierarchy.dendrogram(max_analysis, labels = data['Name'].to_list(), orientation='right')
plt.show()

### (10 points) Group Average analysis ###
average_analysis = hierarchy.average(NumericalAttributes)

# (5 points) Plot the associated dendrogram. 
# Hint1: Make sure each data point is labeled properly (i.e. use argument: labels=data['Name'].tolist())
# Hint2: You can change the orientation of the dendrogram to easily read the labels: orientation='right'
dn = hierarchy.dendrogram(average_analysis, labels = data['Name'].to_list(), orientation='right')
plt.show()
