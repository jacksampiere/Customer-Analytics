import pandas as pd
pd.set_option('display.max_columns', None) # show all columns
pd.set_option('display.float_format', lambda x: '%.2f' % x) # suppress scientific notation
import numpy as np
from scipy.optimize import curve_fit

# read in consensus data
delphi = pd.read_excel('delphi-consensus-outputs.xlsx', index_col=0)
# read in margin and revenue data
margin_rev = pd.read_excel('margin-revenue-salesforce.xlsx', index_col=0)

print('PART A----------------------------------------------------------------------------------------------------------')
# fit nonlinear regression for all drugs

# initialize function to pass into curve_fit()
def sales_response(SF, c, d, adbudg_min, adbudg_max):
	# function to compute sales response to some input salesforce SF
	return adbudg_min + (adbudg_max - adbudg_min) * ((SF**c) / (d + SF**c))

# we'll use 10x the current salesforce size as a proxy for infinity
infinity = 10
# initialize X (same for all drugs)
X = [0, 0.5, 1, 1.5, infinity]
# dictionaries to store parameter values
c_dict, d_dict = {}, {}
min_dict, max_dict = {}, {}
# fit nonlinear regression for each drug
for drug in delphi.columns:
	# extract sales data for the current drug
	y = delphi[drug].values
	# initialize starting vector for the current drug
	p0 = [1, 1, np.min(y), np.max(y)]
	# fit nonlinear regression
	p_opt, p_cov = curve_fit(sales_response, X, y, p0=p0)
	# extract and store parameters c and d
	c, d = p_opt[0], p_opt[1]
	c_dict[drug], d_dict[drug] = c, d
	# extract parameters adbudg_min and adbudg_max
	adbudg_min, adbudg_max = p_opt[2], p_opt[3]
	min_dict[drug], max_dict[drug] = adbudg_min, adbudg_max
	# display drug name and corresponding parameters
	print(f'Optimal parameters for {drug}: c = {c:.2f}, d = {d:.2f}, min = {adbudg_min:.2f}, max = {adbudg_max:.2f}')

print()

print('PART B----------------------------------------------------------------------------------------------------------')
# determine optimal salesforce size for each drug
import scipy.optimize as optimize

# define function to compute negative profit (so optimize.minimize will then maximize profit)
def neg_profit(X, *args):
	# a function to compute the negatve profit of a given salesforce response
	# X = 1D array containing SF (salesforce volume)
	# args = tuple containing c, d, adbudg_min, adbug_max (parameters used to calculate salesforce response), margin, orig_rev, and orig_sf

	# unpack parameters from args tuple
	c, d, adbudg_min, adbudg_max, margin, orig_rev, orig_sf = args

	# extract SF from X
	SF = X[0]
	# compute relative salesforce size
	rel_SF = SF / orig_sf
	# compute sales response
	response = adbudg_min + (adbudg_max - adbudg_min) * ((rel_SF**c) / (d + rel_SF**c))
	# translate to new revenue
	new_rev = response * orig_rev
	# compute profit assuming cost per person is 0.057
	profit = (new_rev * margin) - (SF * 0.057)
	# return negative profit
	return (-1) * profit

# intialize lower bound for optimization
lower_bound = 0
# set bounds via bounds object
bounds = optimize.Bounds(lower_bound, np.inf)
# initial x vector
x0 = [80]
# list to store optimal salesforce volumes
opt_SF = []
# list to store optimal profit values
opt_profit = []

# loop through all drugs and optimize
for drug in delphi.columns:

	# grab info for current drug

	# extract c and d values
	c, d = c_dict[drug], d_dict[drug]
	# extract min and max values
	adbudg_min, adbudg_max = min_dict[drug], max_dict[drug]
	# grab margin
	margin = margin_rev[drug].loc['Profit Margin']	
	# extract original revenue and original salesforce size
	orig_rev, orig_sf = margin_rev[drug].loc['Current/Original Revenue'], margin_rev[drug].loc['Current/Original Salesforce']
	# pack args
	args = (c, d, adbudg_min, adbudg_max, margin, orig_rev, orig_sf)

	# optimize
	optimizer_res = optimize.minimize(neg_profit, args=args, x0=x0, method='trust-constr', bounds=bounds)
	# store optimal salesforce volume
	opt_SF.append(optimizer_res.x[0])
	# store optimal profit
	opt_profit.append((-1) * optimizer_res.fun)
	# display optimal quantities
	print(f'{drug}: Optimal salesforce size = {optimizer_res.x[0]:.2f}, optimal profit = {(-1) * optimizer_res.fun:.2f}')

print()

print('PART C----------------------------------------------------------------------------------------------------------')
# determine optimal salesforce size for each drug subject to having only 700 employees for all drugs combined

def neg_profit_multi(SF, *args):
	# a function to compute the collective negatve profit of a given salesforce response (broadcasted across all drugs)
	# SF = 1D array containing salesforce volume for all drugs
	# args = tuple of vectors containing c, d, adbudg_min, adbug_max (parameters used to calculate salesforce response), margin, orig_rev, and orig_sf for all drugs

	# unpack parameter vectors from args tuple
	c, d, adbudg_min, adbudg_max, margin, orig_rev, orig_sf = args

	# compute relative salesforce size
	rel_SF = SF / orig_sf
	# compute sales response
	response = adbudg_min + (adbudg_max - adbudg_min) * ((rel_SF**c) / (d + rel_SF**c))
	# translate to new revenue
	new_rev = response * orig_rev
	# compute profit assuming cost per person is 0.057
	profit = (new_rev * margin) - (SF * 0.057)
	# sum across all drugs and return negative profit
	return (-1) * np.sum(profit)

# create array of c values
c_arr = np.array(list(c_dict.values()))
# create array of d values
d_arr = np.array(list(d_dict.values()))
# create array of adbudg_min values
min_arr = np.array(list(min_dict.values()))
# create array of adbudg_max values
max_arr = np.array(list(max_dict.values()))
# create array of margins
margin_arr = margin_rev.loc['Profit Margin'].values
# create array of original revenue values
orig_rev_arr = margin_rev.loc['Current/Original Revenue'].values
# create array of original salesforce values
orig_sf_arr = margin_rev.loc['Current/Original Salesforce'].values
# pack into tuple
args_arr = (c_arr, d_arr, min_arr, max_arr, margin_arr, orig_rev_arr, orig_sf_arr)

# number of drugs
n_drugs = 8
# upper bound on salesforce size
total_salesforce_size = 700
# initialize lower bound for optimization
lower_bound = 0
# initial x vector (start with equal salesforces for all drugs)
x0 = np.ones(n_drugs)*total_salesforce_size/n_drugs
# constraint on total salesforce employees across all drugs
sum_constraint_object = optimize.LinearConstraint(np.ones((1,n_drugs)), lower_bound, total_salesforce_size)
# set bounds via bounds object
bounds_object = optimize.Bounds(lower_bound, np.inf)
# optimize
optimizer_res_multi = optimize.minimize(neg_profit_multi, args=args_arr, x0=x0, method='SLSQP', bounds=bounds_object, constraints=sum_constraint_object)

# create dataframe with constrained optima and percent percent_reduction_SF from unconstrained optima

# optimal salesforce volumes in constrained context
opt_SF_multi = optimizer_res_multi.x
# percent reduction of salesforce
percent_reduction_SF = ((np.array(opt_SF) - opt_SF_multi) / np.array(opt_SF)) * 100
# optimal profits in constrained context (have to recalculate since neg_profit_multi returns a scalar for optimization purposes)
opt_profit_multi = [(-1) * neg_profit([opt_SF_multi[i]], c_arr[i], d_arr[i], min_arr[i], max_arr[i], margin_arr[i], orig_rev_arr[i], orig_sf_arr[i]) for i in range(n_drugs)]
# percent reduction of profit
percent_reduction_profit = ((np.array(opt_profit) - opt_profit_multi) / np.array(opt_profit)) * 100


# create dictionary as a precursor to dataframe
stage_dict = {
	'Optimal SF (constrained)':opt_SF_multi,
	'Percent reduction (SF)': percent_reduction_SF,
	'Optimal profit (constrained)':opt_profit_multi,
	'Percent reduction (profit)':percent_reduction_profit
}

# create dataframe, set drug names as index
df = pd.DataFrame(stage_dict, index=delphi.columns)
# display
print(df)