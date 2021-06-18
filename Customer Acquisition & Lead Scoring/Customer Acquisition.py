import pandas as pd
pd.set_option('display.max_columns', None) # show all columns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# read estimation data
est_df = pd.read_excel('prospectscoringhw.xlsx', skiprows=2, skipfooter=304)

# read holdout data
hold_df = pd.read_excel('prospectscoringhw.xlsx', skiprows=206)

print('Q1------------------------------------------------------------------------------')

# pull X and y variables from estimation data
X_est = est_df.drop(columns=['y'])
y_est = est_df['y']

# initialize and fit logistic regression
lr = LogisticRegression(penalty='none', max_iter=1000)
lr.fit(X_est, y_est)
# display intercept
print(f'Intercept: {lr.intercept_[0]:.2f}')

# display coefficients
cols, coeffs = est_df.columns, lr.coef_[0]
print('Coefficients:')
for i in range(len(coeffs)):
	print(f'\t{cols[i]}: {coeffs[i]:.4f}')

print()

print('Q2------------------------------------------------------------------------------')

# pull X and y variables from holdout data
X_hold = hold_df.drop(columns=['y'])
y_hold = hold_df['y']
# calculate scores (i.e. t = beta_0 + beta_1 x gender x ...)
score_arr = lr.decision_function(X_hold)
# predict on holdout set
prob_arr = lr.predict_proba(X_hold)
# extract probability of conversion (i.e. probability of 1)
response_prob = prob_arr[:,1]
# compute lift, dividing by mean response rate in the estimation set
lift = response_prob / (y_est.sum() / len(y_est))

print()

print('Q3------------------------------------------------------------------------------')

# sort descending
lift_sorted = np.flip(np.sort(lift))

print()

print('Q4------------------------------------------------------------------------------')

# sort response probabilities
response_prob_sorted = np.flip(np.sort(response_prob))
# initizlize number of solicitations
num_solicitations = [i for i in range(len(hold_df))]
# plot marginal response rate vs. number of solicitations
plt.plot(num_solicitations, response_prob_sorted)
plt.xlabel('Number of solicitations (n)')
plt.ylabel('Nth Best Response Rate')
plt.show()

print()

print('Q5------------------------------------------------------------------------------')

# define LT customer equity
lt_cust_eq = 30
# define solicitation cost
solicitation_cost = 12
# calculate the max probability of conversion that should be contacted
p_n = solicitation_cost / lt_cust_eq
# subset probabilities to retain only those with r > p_n
contact_arr = response_prob[response_prob >= p_n]
# display number of prospects to contact
print(len(contact_arr))

print()

print('Q6------------------------------------------------------------------------------')

# compute running sum array of response probabilites
running_sum = np.cumsum(response_prob_sorted)
# plot running sum vs. number of solicitations
plt.plot(num_solicitations, running_sum)
plt.xlabel('Number of solicitations (n)')
plt.ylabel('Expected Total Number of Responses')
plt.show()

# separate estimation data by classes
est_df1 = est_df[est_df['y'] == 1]
est_df0 = est_df[est_df['y'] == 0]
# display histograms of good prospects
est_df1.hist(figsize=(6,8), xlabelsize=4, ylabelsize=4)
plt.suptitle('Good Prospects, Melrose Chocolate House')
plt.show()
# display histograms of bad prospects
est_df0.hist(figsize=(6,8), xlabelsize=4, ylabelsize=4)
plt.suptitle('Bad Prospects, Melrose Chocolate House')
plt.show()

# load data from lecture example
est_oth_df = pd.read_excel('acquisition.xlsx', sheet_name='List Data', skiprows=2, skipfooter=204)
# separate estimation data by classes
est_oth_df1 = est_oth_df[est_oth_df['y'] == 1]
est_oth_df0 = est_oth_df[est_oth_df['y'] == 0]
# display histograms of good prospects
est_oth_df1.hist(figsize=(4,5), xlabelsize=4, ylabelsize=4)
plt.suptitle('Good Prospects, Lecture Example')
plt.show()
# display histograms of bad prospects
est_oth_df0.hist(figsize=(4,5), xlabelsize=4, ylabelsize=4)
plt.suptitle('Bad Prospects, Lecture Example')
plt.show()

print()

print('Q7------------------------------------------------------------------------------')

# limited supply
max_boxes = 40
# find first index that is over max_boxes
num_contacts = np.argmax(running_sum > 40)
print(num_contacts)

print()

print('Q8------------------------------------------------------------------------------')

# join response probabilites with true labels
validation = np.column_stack((response_prob, y_hold.to_numpy()))
# sort by response probabilities, keeping true labels attached
validation_sorted = validation[np.flip(validation[:,0].argsort())]
# compute running sum for labels
running_sum_val = np.cumsum(validation_sorted, axis=0)
# plot running sum of predicted response probabilities
plt.plot(num_solicitations, running_sum_val[:,0], label='Predicted')
# overlay running sum of validation labels
plt.plot(num_solicitations, running_sum_val[:,1], 'o', label='True')
plt.xlabel('Number of solicitations (n)')
plt.ylabel('Nth Best Response Rate')
plt.legend()
plt.show()