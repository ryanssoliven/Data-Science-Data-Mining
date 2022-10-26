# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 1) (10 points) Load the data from the file 'dataOutliers.npy'
data = np.load("dataOutliers.npy")

# 2) (10 points) Create a scatter plot to visualize the data (This is just a FYI, make sure to comment the below line after you visualized the data)
plt.scatter(data[:,0], data[:,1])
plt.show()

# 3) (50 points) Anomaly detection: Density-based
# Fit the LocalOutlierFactor model for outlier detection
# Then predict the outlier detection labels of the data points
clf = LocalOutlierFactor()
y_pred = clf.fit_predict(data)
n_errors = (y_pred != data[:,1]).sum()
X_scores = clf.negative_outlier_factor_

# 4) (30 points) Plot results: make sure all plots/images are closed before running the below commands
# Create a scatter plot of the data (exact same as in 2) )
# Then, indicate which points are outliers by plotting circles around the outliers
lofs_index = np.where(y_pred!=1)
values = data[lofs_index]
plt.title("Local Outlier Factor (LOF)")
plt.scatter(data[:,0], data[:,1], label="Normal")
plt.scatter(values[:,0],values[:,1], s=100, edgecolors="r", facecolors="none",label="Outlier")
plt.legend(loc="upper left")
plt.show()
