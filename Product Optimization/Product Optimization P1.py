import pandas as pd
pd.set_option('display.max_columns', None)
pd.reset_option('max_columns')
import numpy as np
import matplotlib.pyplot as plt

##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
### FUNCTION DEFINITIONS ###



def compute_utility(df, feat_vec, new_col_name):
	# a function to compute the utility of a product
	# df = a dataframe with the structure of the provided file
	# price_col,..., contain_col = the column name that corresponds to the value of each attribute for product 1

	# extract attribute values
	price = feat_vec[0]
	ins_time= feat_vec[1]
	capacity = feat_vec[2]
	clean = feat_vec[3]
	containment = feat_vec[4]
	brand = feat_vec[5]

	# dictionary to link attribute values to column names
	price_link = {30:'pPr30', 10:'pPr10', 5:'pPr05'}
	ins_time_link = {0.5:'pIn0.5', 1:'pIn1', 3:'pIn3'}
	capacity_link = {12:'pCp12', 20:'pCp20', 32:'pCp32'}
	clean_lev_link = {'Difficult':'pClD', 'Fair':'pClF', 'Easy':'pClE'}
	clean_time_link = {7:'pClD', 5:'pClF', 2:'pClE'}
	contain_link = {'Slosh resistant':'pCnSl', 'Spill resistant':'pCnSp', 'Leak resistant':'pCnLk'}
	brand_link = {'A':'pBrA', 'B':'pBrB', 'C':'pBrC'}

	# link values to column names
	price_col = price_link[price]
	ins_col = ins_time_link[ins_time]
	cap_col = capacity_link[capacity]
	clean_col = clean_lev_link[clean]
	contain_col = contain_link[containment]
	brand_col = brand_link[brand]

	# compute utility
	df[new_col_name] = (df[price_col]*df['IPr']) + (df[ins_col]*df['Iin']) + (df[cap_col]*df['ICp']) + (df[clean_col]*df['ICl']) + \
		(df[contain_col]*df['Icn']) + (df[brand_col]*df['IBr'])

	# return the dataframe
	return df



def logit_adjustment(df, c):
	# implements the logit adjustment of predicted utilities
	# df = a datagrame with columns U1, U2, and U3 (utilities)
	# c = the uncertainty of the utilities

	df['Pr(Product 1)'] = (np.exp(c*df['U1'])) / (np.exp(c*df['U1']) + np.exp(c*df['U2']) + np.exp(c*df['U3']))
	df['Pr(Product 2)'] = (np.exp(c*df['U2'])) / (np.exp(c*df['U1']) + np.exp(c*df['U2']) + np.exp(c*df['U3']))
	df['Pr(Product 3)'] = (np.exp(c*df['U3'])) / (np.exp(c*df['U1']) + np.exp(c*df['U2']) + np.exp(c*df['U3']))

	# return the dataframe with logit-adjusted probabilities
	return df



def create_eba_df(this_cust, vec_a, vec_b, vec_c):
    
    # dictionaries to link product attributes to column names
    price_link = {30:'pPr30', 10:'pPr10', 5:'pPr05'}
    ins_time_link = {0.5:'pIn0.5', 1:'pIn1', 3:'pIn3'}
    capacity_link = {12:'pCp12', 20:'pCp20', 32:'pCp32'}
    clean_lev_link = {'Difficult':'pClD', 'Fair':'pClF', 'Easy':'pClE'}
    clean_time_link = {7:'pClD', 5:'pClF', 2:'pClE'}
    contain_link = {'Slosh resistant':'pCnSl', 'Spill resistant':'pCnSp', 'Leak resistant':'pCnLk'}
    brand_link = {'A':'pBrA', 'B':'pBrB', 'C':'pBrC'}
    
    # create lists of values for each product
    price_col = [this_cust[price_link[vec_a[0]]].values[0],
                 this_cust[price_link[vec_b[0]]].values[0],
                 this_cust[price_link[vec_c[0]]].values[0]]
    ins_time_col = [this_cust[ins_time_link[vec_a[1]]].values[0],
                 this_cust[ins_time_link[vec_b[1]]].values[0],
                 this_cust[ins_time_link[vec_c[1]]].values[0]]
    cap_col = [this_cust[capacity_link[vec_a[2]]].values[0],
                 this_cust[capacity_link[vec_b[2]]].values[0],
                 this_cust[capacity_link[vec_c[2]]].values[0]]
    clean_lev_col = [this_cust[clean_lev_link[vec_a[3]]].values[0],
                 this_cust[clean_lev_link[vec_b[3]]].values[0],
                 this_cust[clean_lev_link[vec_c[3]]].values[0]]
    cont_col = [this_cust[contain_link[vec_a[4]]].values[0],
                 this_cust[contain_link[vec_b[4]]].values[0],
                 this_cust[contain_link[vec_c[4]]].values[0]]
    brand_col = [this_cust[brand_link[vec_a[5]]].values[0],
                 this_cust[brand_link[vec_b[5]]].values[0],
                 this_cust[brand_link[vec_c[5]]].values[0]]
    
    # organize into dataframe
    for_prod_df = {
        'Price':price_col, 'Ins. Time':ins_time_col, 'Capacity':cap_col,
        'Clean Level':clean_lev_col, 'Containment':cont_col, 'Brand':brand_col
    }

    # export product info to dataframe
    prod_df = pd.DataFrame(for_prod_df, index = ['A', 'B', 'C'])

    return prod_df



def implement_eba(ratings, importances, cutoffs):
	# implements elimination by aspects
    
    # stack importances onto ratings array
    stacked = np.insert(ratings, 0, importances, axis=0)
    
    # randomly shuffle columns (for randomizing equally important attributes)
    stacked = stacked[:, np.random.permutation(stacked.shape[1])]
    
    # sort based on importances and slice off just the ratings
    sorted_arr = stacked[:, np.flip(np.argsort(stacked[0,:], ))][1:]

    # grab the indices of the unsatisfied characteristics for each row
    bool_arr = list(map(lambda x: np.where((x > cutoffs) == False)[0], sorted_arr))
    
    # get the first index of the unsatisfied characteristic, set to 6 if everything is satisfied
    choice_vec = np.fromiter(map(lambda x: x[0] if len(x) > 0 else len(importances), bool_arr), dtype=int)
    
    # return whether the product is retained for the max amount of elimination iterations
    is_max = np.array(choice_vec == choice_vec.max(), dtype=int)

    # return proportion of time that product C is chosen
    return (is_max[-1] / sum(is_max))



##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
### END FUNCTION DEFINITIONS ###



print('QUESTION 1-----------------------------------------------------------------------------------------------------------')
# proposed market scenario
# Brand A, $30, 3 hrs, 20 oz, Clean Easy, Leak Resistant
# Brand B: $10, 1 hrs, 20 oz, Clean Fair, Spill Resistant
# Brand C (our candidate): $30, 1 hrs, 20 oz, Clean Easy, Leak Resistant

# want to find share, price, margin, and expected profit per person with our current candidate
feat_vec_a1 = [30, 3, 20, 'Easy', 'Leak resistant', 'A']
feat_vec_b1 = [10, 1, 20, 'Fair', 'Spill resistant', 'B']
feat_vec_c1 = [30, 1, 20, 'Easy', 'Leak resistant', 'C']

# read data
df_q1 = pd.read_csv('mugs-preference-parameters-full.csv')

# compute product utilities
df_q1 = compute_utility(df_q1, feat_vec_a1, 'U1')
df_q1 = compute_utility(df_q1, feat_vec_b1, 'U2')
df_q1 = compute_utility(df_q1, feat_vec_c1, 'U3')

# generate probabilities via logit adjustment
c = 0.0139
df_q1 = logit_adjustment(df_q1, c)

# derive market share of our candidate
market_share_c1 = df_q1['Pr(Product 3)'].mean()
print(f'Share: {market_share_c1*100:.2f}%')
# we are given the price
price_c1 = 30
print(f'Price: ${price_c1:.2f} (given)')

# cost structure of our candidate
ins_cost_c1 = 1
cap_cost_c1 = 2.6
clean_cost_c1 = 3
cont_cost_c1 = 1
# total cost of our candidate
total_cost_c1 = ins_cost_c1 + cap_cost_c1 + clean_cost_c1 + cont_cost_c1
# margin of our candidate
margin_c1 = price_c1 - total_cost_c1
print(f'Margin: ${margin_c1:.2f}')
# profit per person
prof_per_person_c1 = market_share_c1 * margin_c1
print(f'Profit per person: ${prof_per_person_c1:.2f}')

print()

print('QUESTION 2-----------------------------------------------------------------------------------------------------------')
# features over which we will iterate to generate combinations
prices = [30, 10, 5]
ins_times = [0.5, 1, 3] # this is in hours
capacities = [12, 20, 32]
clean_levels = ['Difficult', 'Fair', 'Easy']
containment_types = ['Slosh resistant', 'Spill resistant', 'Leak resistant']

# dictionaries to link products to cost structure
time_ins_costs = {0.5:0.5, 1:1, 3:3}
cap_costs = {12:1, 20:2.60, 32:2.80}
clean_costs = {'Difficult':1, 'Fair':2.20, 'Easy':3}
cont_costs = {'Slosh resistant':0.5, 'Spill resistant':0.8, 'Leak resistant':1}

# only dealing with our own candidate
brand = 'C'
# always use this c-value for logit adjustment
c = 0.0139

# feature vectors for products from Brand A and Brand B
feat_vec_a2 = [30, 3, 20, 'Easy', 'Leak resistant', 'A']
feat_vec_b2 = [10, 1, 20, 'Fair', 'Spill resistant', 'B']

# lists in which we will store results
length = 3**5
price_results = [0 for i in range(length)]
ins_time_results = [0 for i in range(length)]
cap_results = [0 for i in range(length)]
clean_results = [0 for i in range(length)]
cont_results = [0 for i in range(length)]
share_results = [0 for i in range(length)]
margin_results = [0 for i in range(length)]
expec_prof_results = [0 for i in range(length)]

# looping through all combinations
i = 0
for price in prices:
	for ins_time in ins_times:
		for cap in capacities:
			for clean_level in clean_levels:
				for cont_type in containment_types:
					# track index
					i += 1

					# reinitialize dataframe for each combination
					df_q2 = pd.read_csv('mugs-preference-parameters-full.csv')

					# feature vector for this iteration
					feat_vec_c2 = [price, ins_time, cap, clean_level, cont_type, brand]

					# compute utilities
					df_q2 = compute_utility(df_q2, feat_vec_a2, 'U1')
					df_q2 = compute_utility(df_q2, feat_vec_b2, 'U2')
					df_q2 = compute_utility(df_q2, feat_vec_c2, 'U3')

					# generate probabilities via logit adjustment
					df_q2 = logit_adjustment(df_q2, c)


					# derive market share of our candidate
					market_share_c2 = df_q2['Pr(Product 3)'].mean()
					# cost structure of our candidate
					ins_cost_c2 = time_ins_costs[ins_time]
					cap_cost_c2 = cap_costs[cap]
					clean_cost_c2 = clean_costs[clean_level]
					cont_cost_c2 = cont_costs[cont_type]
					# total cost of our candidate
					total_cost_c2 = ins_cost_c2 + cap_cost_c2 + clean_cost_c2 + cont_cost_c2
					# we know the price
					price_c2 = price
					# margin of our candidate
					margin_c2 = price_c2 - total_cost_c2
					# profit per person
					prof_per_person_c2 = market_share_c2 * margin_c2

					# track and store values
					price_results[i-1] = price_c2
					ins_time_results[i-1] = ins_time
					cap_results[i-1] = cap
					clean_results[i-1] = clean_level
					cont_results[i-1] = cont_type
					share_results[i-1] = market_share_c2
					margin_results[i-1] = margin_c2
					expec_prof_results[i-1] = prof_per_person_c2

# create dictionary with result lists
results_data_q2 = {
	'Index': [i for i in range(1,244)],
	'Price': price_results,
	'Time Insulated': ins_time_results,
	'Capacity': cap_results,
	'Cleanability': clean_results,
	'Containment': cont_results,
	'Market Share': share_results,
	'Margin': margin_results,
	'Expected Profit per Person': expec_prof_results
}

# create dataframe of results
df_results_q2 = pd.DataFrame(results_data_q2)

# modify columns and export results to csv
csv_q2_df = df_results_q2.copy()
csv_q2_df['Price'] = '$' + csv_q2_df['Price'].astype(str)
csv_q2_df['Time Insulated'] = csv_q2_df['Time Insulated'].astype(str) + ' hrs'
csv_q2_df['Capacity'] = csv_q2_df['Capacity'].astype(str) + ' oz'
csv_q2_df['Market Share'] = (csv_q2_df['Market Share']*100).astype(str) + '%'
csv_q2_df['Margin'] = '$' + csv_q2_df['Margin'].astype(str)
csv_q2_df['Expected Profit per Person'] = '$' + csv_q2_df['Expected Profit per Person'].astype(str)

csv_q2_df.to_csv('results_q2.csv', header=True, index=False)

# plot share (x) vs. expected profit per person (y)
plt.plot(df_results_q2['Market Share'], df_results_q2['Expected Profit per Person'], 'o')
plt.xlabel('Market Share')
plt.ylabel('Expected Profit per Capita')
plt.title('Profit per Capita vs. Market Share')
plt.show()

print()

print('QUESTION 3-----------------------------------------------------------------------------------------------------------')
# extract index of candidate with highest profit
idx_optimal = df_results_q2['Expected Profit per Person'].idxmax()

# extract each attribute
price_q3 = df_results_q2['Price'].iloc[idx_optimal]
ins_q3 = df_results_q2['Time Insulated'].iloc[idx_optimal]
cap_q3 = df_results_q2['Capacity'].iloc[idx_optimal]
clean_q3 = df_results_q2['Cleanability'].iloc[idx_optimal]
cont_q3 = df_results_q2['Containment'].iloc[idx_optimal]
share_q3 = df_results_q2['Market Share'].iloc[idx_optimal]
margin_q3 = df_results_q2['Margin'].iloc[idx_optimal]
expec_prof_q3 = df_results_q2['Expected Profit per Person'].iloc[idx_optimal]

# display results
print(f'Price: ${price_q3:.2f}')
print(f'Time Insulated: {ins_q3} hrs')
print(f'Capacity: {cap_q3} oz')
print(f'Cleanability: {clean_q3}')
print(f'Containment: {cont_q3}')
print(f'Share: {share_q3*100:.2f}%')
print(f'Margin: ${margin_q3:.2f}')
print(f'Expected Profit per Person: ${expec_prof_q3:.2f}')

print()

print('QUESTION 4---------------------------------------------------------------------------------------------------------')
# want to find share, price, margin, and expected profit per person with our current candidate
feat_vec_a41 = [30, 3, 20, 'Easy', 'Leak resistant', 'A']
feat_vec_b41 = [10, 1, 20, 'Fair', 'Spill resistant', 'B']
feat_vec_c41 = [30, 1, 20, 'Easy', 'Leak resistant', 'C']

# read data
df_q4_1 = pd.read_csv('mugs-preference-parameters-full.csv')

# features over which we will iterate to generate combinations
prices = [30, 10, 5]
ins_times = [0.5, 1, 3] # this is in hours
capacities = [12, 20, 32]
clean_levels = ['Difficult', 'Fair', 'Easy']
containment_types = ['Slosh resistant', 'Spill resistant', 'Leak resistant']

# dictionaries to link products to cost structure
time_ins_costs = {0.5:0.5, 1:1, 3:3}
cap_costs = {12:1, 20:2.60, 32:2.80}
clean_costs = {'Difficult':1, 'Fair':2.20, 'Easy':3}
cont_costs = {'Slosh resistant':0.5, 'Spill resistant':0.8, 'Leak resistant':1}

# feature vectors for Brand A and Brand B
feat_vec_a4 = [30, 3, 20, 'Easy', 'Leak resistant', 'A']
feat_vec_b4 = [10, 1, 20, 'Fair', 'Spill resistant', 'B']

# lists in which we will store results
length = 3**5
price_results = [0 for i in range(length)]
ins_time_results = [0 for i in range(length)]
cap_results = [0 for i in range(length)]
clean_results = [0 for i in range(length)]
cont_results = [0 for i in range(length)]
share_results = [0 for i in range(length)]
margin_results = [0 for i in range(length)]
expec_prof_results = [0 for i in range(length)]

# initialize index, cutoffs, and number of customers
i = 0
# num_cust = 311
cutoffs = np.array([2.5 for cutoff in range(6)])

# looping through all products
for price in prices:
	for ins_time in ins_times:
		for cap in capacities:
			for clean_level in clean_levels:
				for cont_type in containment_types:

					# track index
					i += 1

					# reinitialize the dataframe
					df_q4_2 = pd.read_csv('mugs-preference-parameters-full.csv')

					# initialize feature vector for this iteration of the product
					feat_vec_c4 = [price, ins_time, cap, clean_level, cont_type, 'C']

					# list in which to store choice (proportions) of each customer
					cust_vals = [0 for j in range(311)]

					# generate a dataframe for all 3 products with the current specification of Brand C for each customer
					for cust_num in df_q4_2['Cust']:

						# initialize count of the amount of customers that choose our product
						num_choose_c = 0

						# extract row of a single customer
						this_cust_df = df_q4_2[df_q4_2['Cust'] == cust_num]

						# extract importances from this_cust_df
						importances = this_cust_df.values[0][-6:]

						# determine if need to randomize choice
						need_100_iter = (len(np.unique(importances)) != len(importances))

						# convert to an array of ratings for the 3 available products
						ratings = np.array(create_eba_df(this_cust_df, feat_vec_a4, feat_vec_b4, feat_vec_c4))

						if need_100_iter:

							# loop 100x
							for _ in range(100):

								# get choice and track values
								num_choose_c += implement_eba(ratings, importances, cutoffs)

							# take average of choices
							num_choose_c /= 100

						else:

							# perform elimination by aspects
							num_choose_c = implement_eba(ratings, importances, cutoffs)

						# store choices for all 311 customers
						cust_vals[cust_num-1] = num_choose_c

					# calculate market share
					market_share_c4 = sum(cust_vals) / len(cust_vals)
					# using dictionary to determine costs
					ins_cost_c4 = time_ins_costs[ins_time]
					cap_cost_c4 = cap_costs[cap]
					clean_cost_c4 = clean_costs[clean_level]
					cont_cost_c4 = cont_costs[cont_type]
					# total cost of our candidate
					total_cost_c4 = ins_cost_c4 + cap_cost_c4 + clean_cost_c4 + cont_cost_c4
					# we know the price
					price_c4 = price
					# margin of our candidate
					margin_c4 = price_c4 - total_cost_c4
					# profit per person
					prof_per_person_c4 = market_share_c4 * margin_c4

					# track and store values
					price_results[i-1] = price
					ins_time_results[i-1] = ins_time
					cap_results[i-1] = cap
					clean_results[i-1] = clean_level
					cont_results[i-1] = cont_type
					share_results[i-1] = market_share_c4
					margin_results[i-1] = margin_c4
					expec_prof_results[i-1] = prof_per_person_c4

print('QUESTION 4.1---------------------------------------------------------------------------------------------------------')

# display results
print(f'Price: ${price_results[44]:.2f}')
print(f'Time Insulated: {ins_time_results[44]} hrs')
print(f'Capacity: {cap_results[44]} oz')
print(f'Cleanability: {clean_results[44]}')
print(f'Containment: {cont_results[44]}')
print(f'Share: {share_results[44]*100:.2f}%')
print(f'Margin: ${margin_results[44]:.2f}')
print(f'Expected Profit per Person: ${expec_prof_results[44]:.2f}')

print()

print('QUESTION 4.2---------------------------------------------------------------------------------------------------------')

# store lists to text files
with open('price_results.txt', 'w') as filehandle:
    for listitem in price_results:
        filehandle.write('%s\n' % listitem)

with open('ins_time_results.txt', 'w') as filehandle:
    for listitem in ins_time_results:
        filehandle.write('%s\n' % listitem)

with open('cap_results.txt', 'w') as filehandle:
    for listitem in cap_results:
        filehandle.write('%s\n' % listitem)

with open('clean_results.txt', 'w') as filehandle:
    for listitem in clean_results:
        filehandle.write('%s\n' % listitem)

with open('cont_results.txt', 'w') as filehandle:
    for listitem in cont_results:
        filehandle.write('%s\n' % listitem)

with open('share_results.txt', 'w') as filehandle:
    for listitem in share_results:
        filehandle.write('%s\n' % listitem)

with open('margin_results.txt', 'w') as filehandle:
    for listitem in margin_results:
        filehandle.write('%s\n' % listitem)

with open('expec_prof_results.txt', 'w') as filehandle:
    for listitem in expec_prof_results:
        filehandle.write('%s\n' % listitem)

# create dictionary with result lists
results_data_q4_2 = {
	'Index': [i for i in range(1,244)],
	'Price': price_results,
	'Time Insulated': ins_time_results,
	'Capacity': cap_results,
	'Cleanability': clean_results,
	'Containment': cont_results,
	'Market Share': share_results,
	'Margin': margin_results,
	'Expected Profit per Person': expec_prof_results
}

# create dataframe of results
df_results_q4_2 = pd.DataFrame(results_data_q4_2)

# modify columns and export results to csv
csv_q4_2_df = df_results_q4_2.copy()
csv_q4_2_df['Price'] = '$' + csv_q4_2_df['Price'].astype(str)
csv_q4_2_df['Time Insulated'] = csv_q4_2_df['Time Insulated'].astype(str) + ' hrs'
csv_q4_2_df['Capacity'] = csv_q4_2_df['Capacity'].astype(str) + ' oz'
csv_q4_2_df['Market Share'] = (csv_q4_2_df['Market Share']*100).astype(str) + '%'
csv_q4_2_df['Margin'] = '$' + csv_q4_2_df['Margin'].astype(str)
csv_q4_2_df['Expected Profit per Person'] = '$' + csv_q4_2_df['Expected Profit per Person'].astype(str)

# plot share (x) vs. expected profit per person (y)
plt.plot(df_results_q4_2['Market Share'], df_results_q4_2['Expected Profit per Person'], 'o')
plt.xlabel('Market Share')
plt.ylabel('Expected Profit per Capita')
plt.title('Profit per Capita vs. Market Share')
plt.show()

print()

print('QUESTION 4.3---------------------------------------------------------------------------------------------------------')

# extract index of candidate with highest profit
idx_optimal = df_results_q4_2['Expected Profit per Person'].idxmax()

# extract each attribute
price_q4_3 = df_results_q4_2['Price'].iloc[idx_optimal]
ins_q4_3 = df_results_q4_2['Time Insulated'].iloc[idx_optimal]
cap_q4_3 = df_results_q4_2['Capacity'].iloc[idx_optimal]
clean_q4_3 = df_results_q4_2['Cleanability'].iloc[idx_optimal]
cont_q4_3 = df_results_q4_2['Containment'].iloc[idx_optimal]
share_q4_3 = df_results_q4_2['Market Share'].iloc[idx_optimal]
margin_q4_3 = df_results_q4_2['Margin'].iloc[idx_optimal]
expec_prof_q4_3 = df_results_q4_2['Expected Profit per Person'].iloc[idx_optimal]

# display results
print(f'Price: ${price_q4_3:.2f}')
print(f'Time Insulated: {ins_q4_3} hrs')
print(f'Capacity: {cap_q4_3} oz')
print(f'Cleanability: {clean_q4_3}')
print(f'Containment: {cont_q4_3}')
print(f'Share: {share_q4_3*100:.2f}%')
print(f'Margin: ${margin_q4_3:.2f}')
print(f'Expected Profit per Person: ${expec_prof_q4_3:.2f}')

print()