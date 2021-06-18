import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
import seaborn as sns

print('PART A, QUESTION 1-------------------------------------------------------------------------------------')

# import mug data
mugs_df = pd.read_excel('mugs-analysis-full-incl-demographics.xlsx', sheet_name='for-cluster-analysis')
# import purchase probabilities
probs_df = pd.read_excel('mugs-analysis-full-incl-demographics.xlsx', sheet_name='mugs-full', skiprows=[i for i in range(1,30)], usecols='A,BD:BF')
# set column names
probs_df.columns = ['Cust', 'P1', 'P2', 'P3']

# join mugs_df and probs_df
mugs_probs_df = mugs_df.merge(probs_df, on='Cust')

# variable that specifies which columns to iterate through
cols_iter = mugs_probs_df.columns[1:-3]

# create columns based on P3 (i.e. Brand C) and each of the characteristics (exclude cust and probabilities)
for col in cols_iter:

	# create new column name
	col_name = 'P3*' + col
	# multiply corresponding columns by P3
	mugs_probs_df[col_name] = mugs_probs_df['P3'] * mugs_probs_df[col]

# separate P3 columns
cols_p3 = [col for col in mugs_probs_df if col.startswith('P3*')]
mugs_probs_p3_df = mugs_probs_df[cols_p3]
# rename columns of p3 df
mugs_probs_p3_df.columns = [col[3:] for col in mugs_probs_p3_df]

# compute sum of purchase probabilities for P3
weight_sum_p3 = mugs_probs_df['P3'].sum()
# get weighted averages
weighted_avg_p3_df = pd.DataFrame(mugs_probs_p3_df.sum(axis=0)).transpose() / weight_sum_p3
# set index name
weighted_avg_p3_df.index = ['P3 Mean']

cols_p3 = weighted_avg_p3_df.columns
vals_p3 = weighted_avg_p3_df.values[0]

for i in range(len(cols_p3)):
	print(f'Weighted average of {cols_p3[i]} for P3 is {vals_p3[i]:.2f}')

# reinitialize mugs_probs_df
mugs_probs_df = mugs_df.merge(probs_df, on='Cust')

print()

print('PART A, QUESTION 2, BRAND A----------------------------------------------------------------------------')

# variable that specifies which columns to iterate through
cols_iter = mugs_probs_df.columns[1:-3]

# create columns based on P3 (i.e. Brand C) and each of the characteristics (exclude cust and probabilities)
for col in cols_iter:

	# create new column name
	col_name = 'P1*' + col
	# multiply corresponding columns by P3
	mugs_probs_df[col_name] = mugs_probs_df['P1'] * mugs_probs_df[col]

# separate P3 columns
cols_p1 = [col for col in mugs_probs_df if col.startswith('P1*')]
mugs_probs_p1_df = mugs_probs_df[cols_p1]
# rename columns of p3 df
mugs_probs_p1_df.columns = [col[3:] for col in mugs_probs_p1_df]

# compute sum of purchase probabilities for P3
weight_sum_p1 = mugs_probs_df['P1'].sum()
# get weighted averages
weighted_avg_p1_df = pd.DataFrame(mugs_probs_p1_df.sum(axis=0)).transpose() / weight_sum_p1
# set index name
weighted_avg_p1_df.index = ['P1 Mean']

cols_p1 = weighted_avg_p1_df.columns
vals_p1 = weighted_avg_p1_df.values[0]

for i in range(len(cols_p1)):
	print(f'Weighted average of {cols_p1[i]} for P1 is {vals_p1[i]:.2f}')

# reinitialize mugs_probs_df
mugs_probs_df = mugs_df.merge(probs_df, on='Cust')

print()

print('PART A, QUESTION 2, BRAND B----------------------------------------------------------------------------')

# variable that specifies which columns to iterate through
cols_iter = mugs_probs_df.columns[1:-3]

# create columns based on P3 (i.e. Brand C) and each of the characteristics (exclude cust and probabilities)
for col in cols_iter:

	# create new column name
	col_name = 'P2*' + col
	# multiply corresponding columns by P3
	mugs_probs_df[col_name] = mugs_probs_df['P2'] * mugs_probs_df[col]

# separate P3 columns
cols_p2 = [col for col in mugs_probs_df if col.startswith('P2*')]
mugs_probs_p2_df = mugs_probs_df[cols_p2]
# rename columns of p3 df
mugs_probs_p2_df.columns = [col[3:] for col in mugs_probs_p2_df]

# compute sum of purchase probabilities for P3
weight_sum_p2 = mugs_probs_df['P2'].sum()
# get weighted averages
weighted_avg_p2_df = pd.DataFrame(mugs_probs_p2_df.sum(axis=0)).transpose() / weight_sum_p2
# set index name
weighted_avg_p2_df.index = ['P2 Mean']

cols_p2 = weighted_avg_p2_df.columns
vals_p2 = weighted_avg_p2_df.values[0]

for i in range(len(cols_p2)):
	print(f'Weighted average of {cols_p2[i]} for P2 is {vals_p2[i]:.2f}')

# reinitialize mugs_probs_df
mugs_probs_df = mugs_df.merge(probs_df, on='Cust')

print()

print('PART A, QUESTION 2, OVERALL MEANS + LOG LIFTS----------------------------------------------------------')

# extract descriptors from mugs_probs_df
descriptors_df = mugs_probs_df[mugs_probs_df.columns[1:-3]]
# convert to overall averages of descriptors
overall_avg_df = pd.DataFrame(descriptors_df.mean(axis=0)).transpose()
# set index name
overall_avg_df.index = ['Overall Mean']

cols_avg = overall_avg_df.columns
vals_avg = overall_avg_df.values[0]

for i in range(len(cols_avg)):
	print(f'Overall average of {cols_avg[i]} is {vals_avg[i]:.2f}')

print()

# creating a dataframe with segment means and overall means
all_segments_df = pd.concat([weighted_avg_p1_df, weighted_avg_p2_df, weighted_avg_p3_df, overall_avg_df], axis=0)

# transpose to simplify the log-lift computations
all_segments_T_df = all_segments_df.transpose()
# construct log lift dataframe
for col in ['P1 Mean', 'P2 Mean', 'P3 Mean']:

	# create name for log-lift column
	col_name = col[:2] + ' LL'
	# create log-lift column
	all_segments_T_df[col_name] = np.log10(all_segments_T_df[col] / all_segments_T_df['Overall Mean'])

everything_df = all_segments_T_df.transpose()
log_lift_df = everything_df.iloc[-3:]
print(log_lift_df)

# set gloabl font size to fit everything on figure
plt.rcParams['axes.labelsize'] = 16
# display heatmap of log-lift df
heatmap = sns.heatmap(log_lift_df, annot=True, center=0, xticklabels=log_lift_df.columns, annot_kws={'size':6}, fmt='.2f')
plt.tight_layout()
plt.show()

print()

print('PART B, K MEANS ANALYSIS-------------------------------------------------------------------------------')
# import mug data
mugs_df = pd.read_excel('mugs-analysis-full-incl-demographics.xlsx', sheet_name='for-cluster-analysis')
# remove demographic variables and convert to numpy for k means
X = np.array(mugs_df.drop(columns=['Cust', 'income', 'age', 'sports', 'gradschl']))

# lists in which to store k values and within-cluster sum of squares
k_vals = [0 for i in range(2,11)]
within_cluster_ss_vals = [0 for i in range(2,11)]

# implement k means
for i, k in enumerate(range(2,11)):

	# set seed for reproducibility
	random.seed(410014)

	# initialize and fit model
	k_means_mod = KMeans(n_clusters=k, n_init=50, max_iter=100)
	k_means_mod.fit(X)

	# retrive within-cluster sum of squares
	within_cluster_ss = k_means_mod.inertia_ / X.shape[0]

	# store values
	k_vals[i] = k
	within_cluster_ss_vals[i] = within_cluster_ss

# plot results
plt.plot(k_vals, within_cluster_ss_vals)
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares')
plt.show()

print()

print('PART B, QUESTION 1-------------------------------------------------------------------------------------')
# set seed for reproducibility
random.seed(410014)
# initialize the model with 4 clusters
k_means_mod = KMeans(n_clusters=4, n_init=50, max_iter=100)
# redefine X to be safe
X = np.array(mugs_df.drop(columns=['Cust', 'income', 'age', 'sports', 'gradschl']))
# fit the model
k_means_mod.fit(X)
# make a copy of mugs_df to store KMeans data
k_means_df = mugs_df.copy()
# assign labels to each data point
k_means_df['Cluster ID'] = k_means_mod.labels_
# average all attributes across clusters, drop customer column
k_means_agg_df = k_means_df.groupby('Cluster ID').mean()
print(k_means_agg_df.drop(columns=['Cust']))

print()

print('PART B, QUESTION 2-------------------------------------------------------------------------------------')
# transpose to simplify log-lift computations
k_means_agg_T_df = k_means_agg_df.transpose()

# compute overall mean
k_means_agg_T_df['Overall Mean'] = mugs_df.drop(columns=['Cust']).mean(axis=0)
# construct log lift dataframe
for col in [i for i in range(4)]:

	# create name for log-lift column
	col_name = 'Seg. ' + str(col)
	# create log-lift column
	k_means_agg_T_df[col_name] = np.log10(k_means_agg_T_df[col] / k_means_agg_T_df['Overall Mean'])

# construct dataframe with just log-lifts
log_lift_k_means_df = k_means_agg_T_df.drop(columns=[0, 1, 2, 3, 'Overall Mean']).transpose()
log_lift_k_means_df.drop(columns=['Cust'], inplace=True)
print(log_lift_k_means_df.head())

# display heatmap of log-lift df
heatmap = sns.heatmap(log_lift_k_means_df, annot=True, center=0, xticklabels=log_lift_k_means_df.columns, annot_kws={'size':5}, fmt='.2f')
plt.tight_layout()
plt.show()