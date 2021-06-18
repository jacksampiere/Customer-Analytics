import pandas as pd
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt
import glob

print('PART A--------------------------------------------------------------------------------------------------------------------------')

# function to pass to curve_fit
def clicks(b, alpha, beta):

	# compute number of clicks given alpha, beta, and bid (b)
	return alpha * (1 - np.exp((-1) * beta * b))

# files to loop through
csvs = ['clicksdata.kw8322228.csv', 'clicksdata.kw8322392.csv', 'clicksdata.kw8322393.csv', 'clicksdata.kw8322445.csv']

# dictionary to store all alpha and beta estimates
estimates = {}

# bounds for alpha and beta (both non-negative)
bounds = ((0,0),(np.inf, np.inf))

# loop through all CSVs
for filename in csvs:

		# read current CSV into data frame
		df_curr = pd.read_csv(filename, index_col=0)

		# extract X and y values from dataframe
		X = df_curr['bid.value'].values
		y = df_curr['n.clicks'].values

		# set initial alpha to max(y)
		alpha0 = max(y)
		# dataframe sorted in order of increasing distance from alpha0/2
		df_curr_sort = df_curr.iloc[(df_curr['n.clicks'] - alpha0/2).abs().argsort()]
		# get the value of b that (roughly) yields alpha0/2
		b_half = df_curr_sort.values[0][0]
		# set initial beta
		beta0 = np.log(2) / b_half
		# pack intial alpha and beta values into initial vector
		p0 = [alpha0, beta0]
		# optimize curve
		p_opt, p_cov = curve_fit(clicks, X, y, p0=p0, bounds=bounds)
		# extact alpha and beta estimates from p_opt
		alpha_est, beta_est = p_opt[0], p_opt[1]

		# pull keyword name from file name
		idx_start, idx_end = filename.rindex('.kw') + 1, filename.rindex('.csv')
		kw = filename[idx_start:idx_end]
		# store alpha and beta estimate in dictionary with file name as key
		estimates[kw] = (alpha_est,beta_est)
		# verify solution by checking residual sum of squares
		rss = np.sum((y - clicks(X, p_opt[0], p_opt[1]))**2)
		print(f'RSS = {rss:.4f}')

# display estimates of alpha and beta
for key, value in estimates.items():
	print(f'Parameters for {key}: alpha = {value[0]:.4f}, beta = {value[1]:.4f}')

print()

print('PART B--------------------------------------------------------------------------------------------------------------------------')

# since we have no budget constraints we can simply do bid optimization for each keyword separately

# read conversion rate data
df_conv = pd.read_excel('hw-kw-ltv-conv.rate-data.xlsx')

# function to pass to minimize
def neg_profit(b, *args):

	# return negative profit given some b (bid), alpha, beta, LTV, and conversion rate
	alpha, beta, LTV, conversion_rate = args
	return (-1) * (alpha * (1 - np.exp((-1) * beta * b)) * (LTV * conversion_rate - b))

# track total profit and total expenditure to check with known values
total_prof, total_exp = 0, 0
# dictionary to store optimal bids
bids = {}

# loop through all keywords
for kw in df_conv['keyword']:

	# pull LTV and conversion rate for current keyword
	LTV = df_conv[df_conv['keyword'] == kw]['ltv'].values[0]
	conversion_rate = df_conv[df_conv['keyword'] == kw]['conv.rate'].values[0]
	# pull alpha and beta from dictionary from part A for the current keyword
	alpha, beta = estimates[kw][0], estimates[kw][1]
	# can't have a negative bid
	bounds = Bounds(0, np.inf)
	# set initial x value(s)
	x0 = [0.001]
	# create tuple of additional arguments to pass into minimize
	args = (alpha, beta, LTV, conversion_rate)

	# optimize
	optimizer_res = minimize(neg_profit, args=args, x0=x0, bounds=bounds)
	# record optimal bid, profit, and expenditure
	b_opt = optimizer_res.x[0]
	profit = (-1) * optimizer_res.fun[0]
	exp = b_opt * (alpha * (1 - np.exp((-1) * beta * b_opt)))

	# store optimal bid, profit, and expenditure
	bids[kw] = (b_opt, profit, exp)
	# track total profit and total expenditure
	total_prof += profit
	total_exp += exp


# display optimal bids, profit, and expenditures
for key, value in bids.items():
	print(f'{key}: optimal bid = ${value[0]:.2f}, optimal profit = ${value[1]:.2f}, optimal expenditure = ${value[2]:.2f}')

# display totals
print(f'Total profit = ${total_prof:.2f}')
print(f'Total expenditure = ${total_exp:.2f}')

print()

print('PART C--------------------------------------------------------------------------------------------------------------------------')

# function to pass to minimize
def neg_profit_multi(b_vals, *args):
	# compute total negative profit given some vector of bids (b), alpha values, beta values , LTVs, and conversion rates

	# extract parameters from args
	alphas, betas, LTVs, conversion_rates = args
	# compute all profits
	profits = alphas * (1 - np.exp((-1) * betas * b_vals)) * (LTVs * conversion_rates - b_vals)
	# return total negative profit 
	return (-1) * np.sum(profits)

# create array of alpha values
alpha_arr = np.array([value[0] for value in estimates.values()])
# create array of beta values
beta_arr = np.array([value[1] for value in estimates.values()])
# create array of LTVs
LTV_arr = df_conv['ltv'].values
# create array of conversion rates
conversion_rate_arr = df_conv['conv.rate'].values
# pack arrays of values into args tuple
args_arr = (alpha_arr, beta_arr, LTV_arr, conversion_rate_arr)

# function to pass to NonLinearConstraint
def total_exp(b_vals):
	# compute total expenditure given some vector of bids

	# Compute number of clicks
	n_clicks = alpha_arr * (1 - np.exp((-1) * beta_arr * b_vals))
	# return total expenditure
	return np.sum(b_vals * n_clicks)

# number of keywords
n_kw = 4
# can't have a negative bid
bounds = Bounds(0, np.inf)
# amount to spend on bids
budget = 3000
# set budget constraint
budget_constr_obj = NonlinearConstraint(total_exp, 0, budget)
# set initial vector
x0 = np.zeros(n_kw) + 0.001
# optimize
optimizer_res_multi = minimize(neg_profit_multi, args=args_arr, method='trust-constr', x0=x0, bounds=bounds, constraints=budget_constr_obj)
# record optimal bids and profit
b_opt_constr = optimizer_res_multi.x
profit_constr = (-1) * optimizer_res_multi.fun
print(f'Objective function value: {profit_constr:.2f}')

# function to calculate expenditure of a single bid
def expenditure(b, alpha, beta):

	# compute number of clicks
	n_clicks = alpha * (1 - np.exp((-1) * beta * b))
	# return expenditure
	return b * n_clicks

# function to compute the profit of a single bid
def profit(b, alpha, beta, LTV, conversion_rate):
	# return profit
	return alpha * (1 - np.exp((-1) * beta * b)) * (LTV * conversion_rate - b)


# track total profit and total expenditure to check with known values
total_prof, total_exp = 0, 0
# dictionary to store optimal constrained bids and expenditures
bids_constr = {}
# loop through all keywords
for i,kw in enumerate(df_conv['keyword']):

	# extract optimal bid for the current keyword
	b_opt_kw = b_opt_constr[i]
	# extract optimal parameters for the current keyword
	alpha_kw, beta_kw = estimates[kw][0], estimates[kw][1]
	# extract LTV and conversion rate for the current keyword
	LTV_kw = df_conv[df_conv['keyword'] == kw]['ltv'].values[0]
	conversion_rate_kw = df_conv[df_conv['keyword'] == kw]['conv.rate'].values[0]
	# compute optimal profit and expenditure
	prof_kw = profit(b_opt_kw, alpha_kw, beta_kw, LTV_kw, conversion_rate_kw)
	exp_kw = expenditure(b_opt_kw, alpha_kw, beta_kw)
	# store optimal bid and expenditure
	bids_constr[kw] = (b_opt_kw, prof_kw, exp_kw)
	# track total profit and total expenditure
	total_prof += prof_kw
	total_exp += exp_kw

# display optimal constrained bids and expenditures
for key, value in bids_constr.items():
	print(f'{key}: constrained bid = ${value[0]:.2f}, constrained profit = ${value[1]:.2f}, constrained expenditure = ${value[2]:.2f}')

# display totals
print(f'Total profit = ${total_prof:.2f}')
print(f'Total expenditure = ${total_exp:.2f}')

print()

print('PART D, E--------------------------------------------------------------------------------------------------------------------------')

# add parameter and bid estimates with data
df_conv['alpha'] = [value[0] for value in estimates.values()]
df_conv['beta'] = [value[1] for value in estimates.values()]
df_conv['b'] = [value[0] for value in bids.values()]
df_conv['b_constrained'] = [value[0] for value in bids_constr.values()]
df_conv['percent reduction'] = (-1) * 100 * (df_conv['b_constrained'] - df_conv['b']) / df_conv['b']

# plot alpha vs. LTV
plt.plot(df_conv['ltv'], df_conv['alpha'], 'o')
plt.xlabel('LTV')
plt.ylabel('alpha')
plt.show()

# plot beta vs. LTV
plt.plot(df_conv['ltv'], df_conv['beta'], 'o')
plt.xlabel('LTV')
plt.ylabel('beta')
plt.show()

# plot bid vs. LTV
plt.plot(df_conv['ltv'], df_conv['b'], 'o')
plt.xlabel('LTV')
plt.ylabel('b')
plt.show()

# plot percent reduction vs. alpha
plt.plot(df_conv['alpha'], df_conv['percent reduction'], 'o')
plt.xlabel('alpha')
plt.ylabel('Percent Reduction')
plt.show()

# plot percent reduction vs. beta
plt.plot(df_conv['beta'], df_conv['percent reduction'], 'o')
plt.xlabel('beta')
plt.ylabel('Percent Reduction')
plt.show()

# plot percent reduction vs. LTV
plt.plot(df_conv['ltv'], df_conv['percent reduction'], 'o')
plt.xlabel('LTV')
plt.ylabel('Percent Reduction')
plt.show()