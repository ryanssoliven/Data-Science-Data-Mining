import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#pd.set_option('display.expand_frame_repr', False)
#pd.set_option('display.max_rows', None)
#I am using python 3.9 IDLE so these are needed to show all columns and rows instead of only 5

# 2) Read the dataset located here 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)

# 3) Assign new headers to the DataFrame
data.columns = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses','Class']

# 4) Drop the 'Sample code number' attribute 
data = data.drop(['Sample code number'],axis=1)

### Missing Values ###

# 5)Convert the '?' to NaN
data = data.replace('?', np.nan)

# 6) Count the number of missing values in each attribute of the data.
print('Number of missing values:')
for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))
    
# 7) Discard the data points that contain missing values
data = data.dropna()
    
### Outliers ###
    
# 8)Draw a boxplot to identify the columns in the table that contain outliers

data['Bare Nuclei'] = data['Bare Nuclei'].astype(str).astype(int)
#need to convert Bare Nuclei column to int in order to draw the boxplot
plot1 = plt.figure(1)
data.boxplot(column = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
               'Normal Nucleoli', 'Mitoses','Class'])

#The attributes with outliers are: 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bland Chromatin',
#'Normal Nucleoli', 'Mitoses'

### Duplicate Data ###

# 9) Check for duplicate instances.
dups = data.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))

# 10) Drop row duplicates
print('Number of rows before discarding duplicates = %d' % (data.shape[0]))
data = data.drop_duplicates()
print('Number of rows after discarding duplicates = %d' % (data.shape[0]))

### Discretization ###

# 11) Plot a 10-bin histogram of the attribute values 'Clump Thickness' distribution
plot2 = plt.figure(2)
data['Clump Thickness'].hist(bins=10)
plt.show()

# 12)Discretize the Clump Thickness' attribute into 4 bins of equal width.
data['Clump Thickness'] = pd.cut(data['Clump Thickness'], 4)
data['Clump Thickness'].value_counts(sort=False)

#print(data['Clump Thickness'].value_counts(sort=False))
#Range of Values and number of records of each category:
#(0.991, 3.25]    131
#(3.25, 5.5]      140
#(5.5, 7.75]       52
#(7.75, 10.0]     126

### Sampling ### 
# 13) Randomly select 1% of the data without replacement. The random_state argument of the function specifies the seed value of the random number generator.
sample = data.sample(frac=0.01, replace=False, random_state=1)
sample
#print(sample)
