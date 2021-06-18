import pandas as pd
pd.set_option('display.max_columns', None) # show all columns
import numpy as np
from scipy.optimize import curve_fit

# read in consensus data
delphi = pd.read_excel('delphi-consensus-outputs.xlsx', index_col=0)
# extract Naprosyn consensus info
naprosyn = delphi['Naprosyn']

# we'll use 10x the current salesforce size as a proxy for infinity
infinity = 10
# initialize X (same for all drugs)
X = [0, 0.5, 1, 1.5, infinity]
# grab y values from Naprosyn data
y = naprosyn.values

# function to pass into curve_fit()
def sales_response(SF, c, d, adbudg_min, adbudg_max):
	# function to compute sales response to some input salesforce SF
	return adbudg_min + (adbudg_max - adbudg_min) * ((SF**c) / (d + SF**c))

# fit parameters to consensus data
theta_hat, varcov_theta_hat = curve_fit(sales_response, X, y)
# quadruple the standard errors of each estimator since low degrees of freedom --> falsely small confidence intervals
varcov_theta_hat *= 4

# set random seed
np.random.seed(4100142)
# draw 500 samples from multivariate normal distribution
samples = np.random.multivariate_normal(theta_hat, varcov_theta_hat, 500)

# function to compute profit
def naprosyn_profit(SF, c, d, adbudg_min, adbudg_max):
	# function to compute profit

	# initialize original salesforce
	orig_sf = 96.8
	# initialize original revenue
	orig_rev = 214.4
	# inialize margin 
	margin = 0.7
	# compute relative salesforce size
	rel_SF = SF / orig_sf
	# compute sales response
	response = adbudg_min + (adbudg_max - adbudg_min) * ((rel_SF**c) / (d + rel_SF**c))
	# translate to new revenue
	new_rev = response * orig_rev
	# compute profit assuming cost per person is 0.057
	profit = (new_rev * margin) - (SF * 0.057)
	# return negative profit
	return profit

# salesforce sizes to consider
sf_values = np.linspace(100, 440, 11)

# dictionary to store [mean, median, std. dev.] of each salesforce size
measures_dict = {}
# for each salesforce size...
for sf in sf_values:

	# list to store profits for this SF value
	profits = []
	# for each sample...
	for sample in samples:

		# extract c and d from the current sample
		c, d = sample[0], sample[1]
		# extract min and max from the current sample
		adbudg_min, adbudg_max = sample[2], sample[3]
		# compute and store profit
		profit = naprosyn_profit(sf, c, d, adbudg_min, adbudg_max)
		profits.append(profit)

	# compute relevant statistics
	stats = [np.mean(profits), np.median(profits), np.std(profits)]
	# store statistics in dictionary
	measures_dict[sf] = stats

# we'll use the common utility function from portfolio optimization: U = E(r) – 0.5 x A x σ2
# in this context, we'll use the mean profit for expected return and std(mean profit) for volatility
# we can also use an average risk aversion coefficient of A = 2.5
# so out final utility function to optimize is U = mean(profit) - 0.5 * 2.5 * σ2
# we'll also compute utility across various levels of risk aversion

# create dataframe of statistics
measures_df = pd.DataFrame(measures_dict, index=['Mean Profit', 'Median Profit', 'Std Dev'])
# calculate utilities in dataframe
measures_df.loc['Utility (Raw)'] = measures_df.loc['Mean Profit'] - 0.5 * 2.5 * measures_df.loc['Std Dev']
measures_df.loc['Utility (Normalized)'] = measures_df.loc['Mean Profit'] - 0.5 * 2.5 * 10 * measures_df.loc['Std Dev']
# display
print(measures_df)