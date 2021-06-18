import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

print('PART A QUESTION 1----------------------------------------------------------------------------------------')

# read in dissimilarity matrix
cars_od = pd.read_csv("cars.dissimilarity.csv", sep=",", index_col=0)

# k values to try
k_vals = [i for i in range(1, 10)]
# list in which to store stresses (i.e. measure of fit)
stress_vals = [0 for i in range(len(k_vals))]

# try out all k values
for i, k in enumerate(k_vals):

	# initialize MDS
	mds = MDS(n_components=k, metric=True, max_iter=1000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
	# fit MDS
	mds_fit_out = mds.fit(cars_od)

	# store results
	stress_vals[i] = mds_fit_out.stress_

plt.plot(k_vals, stress_vals)
plt.xlabel('k')
plt.ylabel('Stress')
plt.title('Stress as a function of k')
plt.show()

print()

print('PART A QUESTION 2----------------------------------------------------------------------------------------')

# set seed to make plot reproducible
np.random.seed(1)
# initialize MDS with k = 2
mds = MDS(n_components=2, metric=True, max_iter=1000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
# fit MDS
mds_fit_out = mds.fit(cars_od)

# extract points from MDS
coords = mds_fit_out.embedding_
# get labels from data
labels = cars_od.columns
# keep labels from overalpping points
x_offset = 0.1

plt.scatter(coords[:,0], coords[:,1])
for i in range(len(coords)):
	x, y = coords[i][0] + x_offset, coords[i][1]
	plt.annotate(text=labels[i], xy=(x,y))
plt.axis('square')
plt.show()