import pandas as pd
pd.set_option('display.max_columns', None) # show all columns of dataframes
import numpy as np
from scipy.stats import beta
from scipy.stats import t

print('QUESTION 1 PART 1-----------------------------------------------------------------------------------------------------------------')

# read clicks data
clicks = pd.read_excel('clicks.dataset.2.xlsx', index_col='ad')

# calculate failures
clicks.loc['failures'] = clicks.loc['exposures'] - clicks.loc['clicks']
# number of draws from each distribution
num_draws = 100_000
# number of ads
num_ads = 5
# baseline values for s0 and f0
s0, f0 = 1, 1
# calculate clicks of posterior distribution
clicks.loc['clicks_posterior'] = clicks.loc['clicks'] + s0
# calculate failures of posterior distribution
clicks.loc['failures_posterior'] = clicks.loc['failures'] + f0

# dataframe to store draws from beta distribution
draws_beta = pd.DataFrame(columns=['ad_' + str(i) for i in range(1,6)])

# for each ad...
for i in range(num_ads):

	# determine s and f for posterior density distribution
	s_val = clicks[i+1].loc['clicks_posterior']
	f_val = clicks[i+1].loc['failures_posterior']

	# create 100,000 draws from beta disribution for current ad
	draw_curr = beta.rvs(a=s_val, b=f_val, size=num_draws)

	# store draws in dataframe
	draws_beta['ad_' + str(i+1)] = draw_curr

# determine max of each row
draws_beta['max'] = draws_beta[['ad_1','ad_2','ad_3', 'ad_4', 'ad_5']].max(axis=1)

# generate is_max columns
for i,col in enumerate(draws_beta.columns[:-1]):

	# create is_max column for the current ad
	draws_beta['is_max_' + str(i+1)] = (draws_beta[col] == draws_beta['max']).astype(int)

# detemine probabilities of each ad
ad_1_prob = draws_beta['is_max_1'].mean()
ad_2_prob = draws_beta['is_max_2'].mean()
ad_3_prob = draws_beta['is_max_3'].mean()
ad_4_prob = draws_beta['is_max_4'].mean()
ad_5_prob = draws_beta['is_max_5'].mean()

# pack into list
probs = [ad_1_prob, ad_2_prob, ad_3_prob, ad_4_prob, ad_5_prob,]

# display results
for i,prob in enumerate(probs):

	print(f'Bayesian posterior probability that ad_{i+1} CTR is the highest: {prob:.4f}')

print()

print('QUESTION 1 PART 2-----------------------------------------------------------------------------------------------------------------')

# read volume data
volume = pd.read_excel('volumes.dataset.2.xlsx', index_col='cust')

# dataframe to store draws from t distribution
draws_t = draws_beta = pd.DataFrame(columns=['ad_' + str(i) for i in range(1,6)])

# for each ad...
for i in range(num_ads):

	# determine mean of current ad
	mean_curr = volume[volume['ad'] == i+1]['volume'].mean()
	# determine std dev of current ad
	stddev_curr = volume[volume['ad'] == i+1]['volume'].std() / len(volume[volume['ad'] == i+1])
	# determine degrees of freedom
	df = len(volume[volume['ad'] == i+1])
	# draw from t distribution
	draw_curr = t.rvs(df=df, size=num_draws)
	# multiple by std dev and add mean
	draw_curr = (draw_curr * stddev_curr) + mean_curr
	# store in dataframe
	draws_t['ad_' + str(i+1)] = draw_curr

# determine max of each row
draws_t['max'] = draws_t[['ad_1','ad_2','ad_3', 'ad_4', 'ad_5']].max(axis=1)
# generate is_max columns
for i,col in enumerate(draws_t.columns[:-1]):

	# create is_max column for the current ad
	draws_t['is_max_' + str(i+1)] = (draws_t[col] == draws_t['max']).astype(int)

# detemine probabilities of each ad
ad_1_prob = draws_t['is_max_1'].mean()
ad_2_prob = draws_t['is_max_2'].mean()
ad_3_prob = draws_t['is_max_3'].mean()
ad_4_prob = draws_t['is_max_4'].mean()
ad_5_prob = draws_t['is_max_5'].mean()

# pack into list
probs = [ad_1_prob, ad_2_prob, ad_3_prob, ad_4_prob, ad_5_prob,]

# display results
for i,prob in enumerate(probs):

	print(f'Bayesian posterior probability that ad_{i+1} volume is the highest: {prob:.4f}')

print()

print('QUESTION 1 PART 3-----------------------------------------------------------------------------------------------------------------')

# datframe to store EVI info
evi_draws = pd.DataFrame(columns=['ad_' + str(i) for i in range(1,6)])

for i in range(num_ads):

	# determine s and f for posterior density distribution
	s_val = clicks[i+1].loc['clicks_posterior']
	f_val = clicks[i+1].loc['failures_posterior']
	# draw from beta distribution
	draw_curr_beta = beta.rvs(a=s_val, b=f_val, size=num_draws)
	# determine degrees of freedom
	df = len(volume[volume['ad'] == i+1])
	# draw from t distribution
	draw_curr_t = t.rvs(df=df, size=num_draws)
	# determine mean of current ad
	mean_curr = volume[volume['ad'] == i+1]['volume'].mean()
	# determine std dev of current ad
	stddev_curr = volume[volume['ad'] == i+1]['volume'].std() / len(volume[volume['ad'] == i+1])
	# multiple by std dev and add mean
	draw_curr_t = (draw_curr_t * stddev_curr) + mean_curr
	# determine product of draws from t and beta distributions
	draw_curr_prod = draw_curr_t * draw_curr_beta
	# store in dataframe
	evi_draws['ad_' + str(i+1)] = draw_curr_prod

# determine max product among all ads
evi_draws['max'] = evi_draws[['ad_1','ad_2','ad_3', 'ad_4', 'ad_5']].max(axis=1)
# generate is_max columns
for i,col in enumerate(evi_draws.columns[:-1]):

	# create is_max column for the current ad
	evi_draws['is_max_' + str(i+1)] = (evi_draws[col] == evi_draws['max']).astype(int)

# determine probabilities of each ad
ad_1_prob = evi_draws['is_max_1'].mean()
ad_2_prob = evi_draws['is_max_2'].mean()
ad_3_prob = evi_draws['is_max_3'].mean()
ad_4_prob = evi_draws['is_max_4'].mean()
ad_5_prob = evi_draws['is_max_5'].mean()

# pack into list
probs = [ad_1_prob, ad_2_prob, ad_3_prob, ad_4_prob, ad_5_prob,]

# display results
for i,prob in enumerate(probs):

	print(f'Bayesian posterior probability that ad_{i+1} EVI is the highest: {prob:.4f}')

print('QUESTION 3------------------------------------------------------------------------------------------------------------------------')

# compute CTR of each ad
click_vals = np.array(clicks.loc['clicks'] / 1000)
# compute mean volume of each ad
mean_vols = np.array([volume[volume['ad'] == i].mean() for i in range(1,6)])[:,1]
# determine sort order of each metric ranking
click_mean_vol = click_vals * mean_vols

# determine and display ad number that is ranked second for CTR*v
print(probs)
rank_probs = np.argsort(probs)
print(np.where(rank_probs == 1)[0])
# determine and display ad number that is ranked second for CTR*MeanVolume
print(click_mean_vol)
rank_click_mean_vol = np.argsort(click_mean_vol)
print(np.where(rank_click_mean_vol == 1)[0])