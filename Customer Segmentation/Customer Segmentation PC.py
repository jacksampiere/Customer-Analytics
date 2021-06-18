import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from numpy import mean, std
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print('PART C-------------------------------------------------------------------------------------------------')
# import mug data
mugs_df = pd.read_excel('mugs-analysis-full-incl-demographics.xlsx', sheet_name='for-cluster-analysis')
# remove demographic variables and convert to numpy for k means
X_df= mugs_df.drop(columns=['Cust', 'income', 'age', 'sports', 'gradschl'])
# convert to numpy
X = np.array(X_df)
# initialize the model with 4 clusters
k_means_mod = KMeans(n_clusters=4, n_init=50, max_iter=100)
# fit the model
k_means_mod.fit(X)
# record labels
labels = k_means_mod.labels_
# initialize PCA
pca_2 = PCA(n_components=2)
# standardize X to convert from correlations instead on variances
X_std = (X - mean(X)) / std(X)
# fit PCA
plot_columns = pca_2.fit_transform(X_std)
# display results
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

print(f'PC1 explains {pca_2.explained_variance_ratio_[0]*100:.2f}% of the variance.')
print('Most important features:')
# form dataframe with PC1 components
pc1_dict = {'Feature': X_df.columns, 'Contribution to PC1': pca_2.components_[0]}
pc1_df = pd.DataFrame(pc1_dict)
pc1_df.sort_values(by='Contribution to PC1', inplace=True, ascending=False)
print(pc1_df.head())
print(pc1_df.tail())

print()

print(f'PC2 explains {pca_2.explained_variance_ratio_[1]*100:.2f}% of the variance.')
print('Most important features:')
# form dataframe with PC1 components
pc2_dict = {'Feature': X_df.columns, 'Contribution to PC2': pca_2.components_[1]}
pc2_df = pd.DataFrame(pc2_dict)
pc2_df.sort_values(by='Contribution to PC2', inplace=True, ascending=False)
print(pc2_df.head())
print(pc2_df.tail())