import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# adjust font size
plt.rcParams.update({'font.size': 6})
from factor_analyzer import FactorAnalyzer

print('PART C QUESTION 1----------------------------------------------------------------------------------------')

# read in attribute ratings
cars_ar = pd.read_csv("cars.ar.csv", sep=",", index_col=0)

# values to try for n_factors
n_factor_vals = [i for i in range(1, 11)]
# list to store goodness of fit values
gof = [0 for i in range(len(n_factor_vals))]

# try all values
for i, n_factor in enumerate(n_factor_vals):

	# intialize factor analyzer
	fa = FactorAnalyzer(n_factors=n_factor, rotation=None)
	# fit factor analyzer
	fa_fit_out = fa.fit(cars_ar)
	# extract communalities
	fa_communalities = fa_fit_out.get_communalities()
	# extract goodness of fit
	fa_gof = sum(fa_communalities)
	# store goodness of fit
	gof[i] = fa_gof
	# extract scores
	fa_scores = fa_fit_out.transform(cars_ar)
	# extract loadings
	fa_factor_loadings = fa_fit_out.loadings_

plt.plot(n_factor_vals, gof)
plt.xlabel('n_factors')
plt.ylabel('GOF')
plt.title('Factor Analysis GOF vs. n_factors')
plt.show()

print('PART C QUESTION 2----------------------------------------------------------------------------------------')

# intialize factor analyzer
fa = FactorAnalyzer(n_factors=2, rotation=None)
# fit factor analyzer
fa_fit_out = fa.fit(cars_ar)
# extract communalities (i.e. intermediate R^2 values for drawing arrows on perceptual map)
fa_communalities = fa_fit_out.get_communalities()
# extract goodness of fit
fa_gof = sum(fa_communalities)
# extract scores
fa_scores = fa_fit_out.transform(cars_ar)
# extract loadings (similar to the regression coefficeint matrix)
fa_factor_loadings = fa_fit_out.loadings_

# store labels for attribute arrows
arrow_labels = cars_ar.columns
# extract beta_x values from loadings
beta_x_vals = fa_factor_loadings[:,0]
# extract beta_x values from loadings
beta_y_vals = fa_factor_loadings[:,1]
# define length scale factor
arrow_length_scaleup = 2.5

# plot attribute arrows
for i in range(len(fa_communalities)):
	beta_x, beta_y = beta_x_vals[i], beta_y_vals[i]
	r2 = fa_communalities[i]
	label = arrow_labels[i]
	origin_x, origin_y = 0, 0
	end_x = arrow_length_scaleup * r2 * beta_x / np.sqrt(beta_x**2 + beta_y**2)
	end_y = arrow_length_scaleup * r2 * beta_y / np.sqrt(beta_x**2 + beta_y**2)
	plt.arrow(origin_x, origin_y, end_x-origin_x, end_y-origin_y, length_includes_head=True, head_width=0.08, head_length=0.0002)

	# hardcoding the movement of a label so they don't overlap
	adjust_y = 0
	if label == 'Common':
		adjust_y -= 0.05

	plt.annotate(text=label, xy=(end_x,end_y+adjust_y))

# store coordinates for perceptual map
coords = fa_scores
# store labels for each point
labels = cars_ar.index
# keep labels from overlapping points
x_offset = 0.075

plt.scatter(coords[:,0], coords[:,1])
# label each point
for i in range(len(coords)):
	x, y = coords[i][0] + x_offset, coords[i][1]
	plt.annotate(text=labels[i], xy=(x,y))
plt.axis('square')
plt.title('Perceptual Map with Attribute Arrows')
plt.show()