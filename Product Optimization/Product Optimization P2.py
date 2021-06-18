import pandas as pd
pd.set_option('display.max_columns', None)
pd.reset_option('max_columns')

# read product data from compensatory model
compens_df = pd.read_csv('results_q2_nums.csv')
# read product data from EBA model
eba_df = pd.read_csv('results_q4_2_nums.csv')

min_market_shares = [i*0.1 for i in range(1,7)]

for share in min_market_shares:

	# subset data based on minimum market share
	this_compens = compens_df[compens_df['Market Share'] >= share]
	this_compens.reset_index(inplace=True)
	this_eba = eba_df[eba_df['Market Share'] >= share]
	this_eba.reset_index(inplace=True)

	# extract index of optimal compensatory candidate
	idx_optimal_compens = this_compens['Expected Profit per Person'].idxmax()
	# extract candidate number
	candidate_compens = this_compens['Index'].iloc[idx_optimal_compens]
	# extract expected profit
	expec_prof_compens = this_compens['Expected Profit per Person'].iloc[idx_optimal_compens]

	# extract index of optimal EBA candidate
	idx_optimal_eba = this_eba['Expected Profit per Person'].idxmax()
	# extract candidate number
	candidate_eba = this_eba['Index'].iloc[idx_optimal_eba]
	# extract expected profit
	expec_prof_eba = this_eba['Expected Profit per Person'].iloc[idx_optimal_eba]

	# display results
	print(f'Market share threshold = {share*100:.2f}%')
	print(f'Compensatory optimum: Candidate {candidate_compens}, expected profit = ${expec_prof_compens:.2f}')
	print(f'EBA optimum: Candidate {candidate_eba}, expected profit = ${expec_prof_eba:.2f}')

print()

# display products with maximum market share for compensatory model
idx_market_share_compens = compens_df['Market Share'].idxmax()
candidate_market_share_compens = compens_df['Index'].iloc[idx_market_share_compens]
max_share_compens = compens_df['Market Share'].iloc[idx_market_share_compens]
print(f'Maximizing compensatory market share: Candidate {candidate_market_share_compens} with a share of {max_share_compens*100:.2f}%')

# display products with maximum market share for EBA model
idx_market_share_eba = eba_df['Market Share'].idxmax()
candidate_market_share_eba = compens_df['Index'].iloc[idx_market_share_eba]
max_share_eba = eba_df['Market Share'].iloc[idx_market_share_eba]
print(f'Maximizing EBA market share: Candidate {candidate_market_share_eba} with a share of {max_share_eba*100:.2f}%')

print()

# derive ratio of market share to cost
compens_df['Cost'] = compens_df['Price'] - compens_df['Margin']
compens_df['Share / Cost'] = compens_df['Market Share'] / compens_df['Cost']
eba_df['Cost'] = eba_df['Price'] - eba_df['Margin']
eba_df['Share / Cost'] = eba_df['Market Share'] / eba_df['Cost']
# find optimal product per this metric for compensatory model
idx_share_cost_compens = compens_df['Share / Cost'].idxmax()
candidate_share_cost_compens = compens_df['Index'].iloc[idx_share_cost_compens]
share_cost_compens = compens_df['Share / Cost'].iloc[idx_share_cost_compens]
print(f'Maximizing compensatory share/cost: Candidate {candidate_share_cost_compens} with a share/cost of {share_cost_compens*100:.2f} %/$')

# find optimal product per this metric for compensatory model
idx_share_cost_eba = eba_df['Share / Cost'].idxmax()
candidate_share_cost_eba = eba_df['Index'].iloc[idx_share_cost_eba]
share_cost_eba = compens_df['Share / Cost'].iloc[idx_share_cost_eba]
print(f'Maximizing EBA share/cost: Candidate {candidate_share_cost_eba} with a share/cost of {share_cost_eba*100:.2f} %/$')