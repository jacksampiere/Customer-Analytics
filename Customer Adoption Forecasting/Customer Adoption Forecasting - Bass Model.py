import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

##################################################################################################
##################################################################################################
##################################################################################################
# FUNCTION DEFINITIONS



def extrap_discrete_bass(df, p, q, M, t):
	# a function to extrapolate a dataframe via discrete bass model to some time t
	# df - a dataframe with columns t, N(t), A(t), and A(t)^2
	# p, q, M - parameters found by (non)linear regression (and possibly also Chain Ratio Method)
	# t - time out to which we will extrapolate

	# populating F(t) and R(t) with place holders for t=1 to t=15
	for i in range(1, len(df)):
		df.loc[i, 'R(t)'] = np.nan
		df.loc[i, 'F(t)'] = np.nan

	# extrapolating
	for i in range(len(df), 30):
		df.loc[i, 't'] = i+1
		df.loc[i, 'A(t)'] = df.loc[i-1, 'N(t)'] + df.loc[i-1, 'A(t)']
		df.loc[i, 'A(t)^2'] = df.loc[i-1, 'A(t)'] ** 2
		df.loc[i, 'R(t)'] = M - df.loc[i, 'A(t)']
		df.loc[i, 'F(t)'] = p + q * (df.loc[i, 'A(t)'] / M)
		df.loc[i, 'N(t)'] = df.loc[i, 'F(t)'] * df.loc[i, 'R(t)']

	# return extrapolated dataframe
	return df


def extrap_cont_bass(df, p, q, M, t):
	# a function to extrapolate a dataframe via continuous bass model to some time t
	# df - a dataframe with columns t, N(t), and A(t)
	# p, q, M - parameters found by (non)linear regression (and possibly also Chain Ratio Method)
	# t - time out to which we will extrapolate

	# populating F(t) and R(t) with place holder for t=1 to t=15
	for i in range(1, len(df)):
		df.loc[i, 'R(t)'] = np.nan
		df.loc[i, 'F(t)'] = np.nan

	for i in range(14, 30):
		t = i+1
		df.loc[i, 't'] = t
		A_t_num = 1 - np.exp((-1)*(p+q)*t)
		A_t_denom = 1 + (q/p)*np.exp((-1)*(p+q)*t)
		A_t = M * (A_t_num/A_t_denom)
		df.loc[i, 'A(t)'] = A_t
		df.loc[i, 'A(t)^2'] = A_t ** 2
		df.loc[i, 'R(t)'] = M - A_t
		df.loc[i, 'F(t)'] = p + q * (A_t/M)
		df.loc[i, 'N(t)'] = df.loc[i, 'F(t)'] * df.loc[i, 'R(t)']

	# return extrapolated dataframe
	return df



# END FUNCTION DEFINITIONS
##################################################################################################
##################################################################################################
##################################################################################################

# read data
df = pd.read_excel('adoptionseries2_with_noise.xlsx')



print('QUESTION 1.1---------------------------------------------')
# generate A(t) and [A(t)]^2 columns
# recall that A(t) = N(t-1) + A(t-1) with A(1) = 0
df.loc[0, 'A(t)'] = 0
for i in range(1, len(df)):
	df.loc[i, 'A(t)'] = df.loc[i-1, 'N(t)'] + df.loc[i-1, 'A(t)']

df['A(t)^2'] = df['A(t)'] ** 2

# store intermediate df for question 1.2
df_2 = df.copy()
# store intermediate df for question 2
df_q2 = df.copy()

# initialize regression model
lin_reg = LinearRegression()
# define X variables, feature names, and Y
X = df.drop(columns=['t', 'N(t)'])
# features = X.columns
Y = df['N(t)']
# fit model
lin_reg.fit(X, Y)

# assign coefficient values
a = lin_reg.intercept_
b, c = lin_reg.coef_[0], lin_reg.coef_[1]
# solve for p, q, M, and N(30)
p = (np.sqrt(b**2 - 4*a*c) - b) / 2
q = (np.sqrt(b**2 - 4*a*c) + b) / 2
M = -q / c

# extrapolate
df = extrap_discrete_bass(df, p, q, M, 30)

# display values
print(f'p = {p:5f}')
print(f'q = {q:5f}')
print(f'M = {M:5f}')
print(f'N(30) = {df.loc[29, "N(t)"]:5f}')
print()



print('QUESTION 1.2---------------------------------------------')
# initialize function to pass into curve_fit()
def compute_N(A_t, p, q):
	# return N(t) as a function of A_t (M = 100)
	return 100*p + (q-p)*A_t - (q/100)*(A_t**2)

# define X and Y
X = df_2['A(t)']
Y = df_2['N(t)']
# run nonlinear regression
p_opt, p_cov = curve_fit(compute_N, X, Y, p0 = [0.02, 0.5])
# extract p and q from optimal parameters
p, q = p_opt[0], p_opt[1]
# display values
print(f'p = {p:5f}')
print(f'q = {q:5f}')
print()



print('QUESTION 1.3---------------------------------------------')
# reinitialize M
M = 100
# extrapolate
df_2 = extrap_discrete_bass(df_2, p, q, M, 30)

# display value
print(f'N(30) = {df_2.loc[29, "N(t)"]:5f}')
print()



print('QUESTION 1.4---------------------------------------------')
# start from initial data since we now have a new way of calculating A(t)
df_4 = pd.read_excel('adoptionseries2_with_noise.xlsx')

# initialize function to pass into curve_fit()
def compute_N_cont(t, p, q):
	# return N(t) as a function of t
	term_1_num = 1 - np.exp((-1)*(p+q)*t)
	term_1_denom = 1 + (q/p)*np.exp((-1)*(p+q)*t)
	term_2_num = 1 - np.exp((-1)*(p+q)*(t-1))
	term_2_denom = 1 + (q/p)*np.exp((-1)*(p+q)*(t-1))
	# M = 100
	return 100*(term_1_num/term_1_denom) - 100*(term_2_num/term_2_denom)

# define X and Y variables
X = df_4['t']
Y = df_4['N(t)']
# run nonlinear regression
p_opt, p_cov = curve_fit(compute_N_cont, X, Y, p0 = [0.02, 0.5])
# extract p and q from optimal parameters
p, q = p_opt[0], p_opt[1]
# display values
print(f'p = {p:5f}')
print(f'q = {q:5f}')
# continue to assume that...
M = 100

# extrapolate
df_4 = extrap_cont_bass(df_4, p, q, M, 30)

# display value
print(f'N(30) = {df_4.loc[29, "N(t)"]:5f}')
print()



print('QUESTION 2---------------------------------------------')
# given
M = 100

# p values to try
p = np.linspace(0.005, 0.008, 4)
# p = np.linspace(0.003, 0.15, 3)

# q values to try
q = np.linspace(0.1, 0.8, 3)
# q = np.linspace(0.07, 0.35, 3)

# generate plots and display
index = 0
for i, p_val in enumerate(p):
	for j, q_val in enumerate(q):

		# reset data for each (p,q) tuple
		df = pd.read_excel('adoptionseries2_with_noise.xlsx')
		df.loc[0, 'A(t)'] = 0
		for k in range(1, len(df)):
			df.loc[k, 'A(t)'] = df.loc[k-1, 'N(t)'] + df.loc[k-1, 'A(t)']

		# extrapolate for each (p,q) tuple
		df = extrap_discrete_bass(df, p_val, q_val, M, 30)

		# create plots
		index +=1
		fig = plt.figure(figsize = (6,3))
		ax = fig.add_subplot()
		ax.plot(df['t'], df['N(t)'], 'o', )
		ax.set_xlabel('t')
		ax.set_ylabel('N(t)')
		ax.set_ylim([0,21])
		ax.set_xlim([0,30])
		ax.set_xticks([0,10,20,30])
		ax.set_xticklabels([0,10,20,30])
		ax.text(x=1, y=1, s=f'p = {p_val:.4f},\nq = {q_val:.2f}')
		plt.show()

print('Code that generates plots has been commented out for convenience of running the script via CLI. See writeup.')
print()


print('QUESTION 3---------------------------------------------')
# this is the data whose curve doesn't deviate too much despite noise
df_true1 = pd.read_excel('adoptionseries1_14.xlsx')
df_noise1 = pd.read_excel('adoptionseries1_with_noise.xlsx')
plt.plot(df_true1['t'], df_true1['N(t)'], 'o', c='green')
plt.plot(df_noise1['t'], df_noise1['N(t)'], 'o', c='red')
plt.legend(['True', 'Noise'])
plt.title('Noisy Data: Results in Low Error')
plt.xlabel('t')
plt.ylabel('N(t)')
plt.show()

# # # this is the data whose curve deviates a lot due to noise
df_true2 = pd.read_excel('adoptionseries2.xlsx')
df_noise2 = pd.read_excel('adoptionseries2_with_noise.xlsx')
plt.plot(df_true2['t'], df_true2['N(t)'], 'o', c='green')
plt.plot(df_noise2['t'], df_noise2['N(t)'], 'o', c='red')
plt.legend(['True', 'Noise'])
plt.title('Noisy Data: Results in High Error')
plt.xlabel('t')
plt.ylabel('N(t)')
plt.show()

# let's extrapolate each series and see what happens


# LOW-ERROR CASE
# populating each dataframe with values for A(t) and A(t)^^2
df_true1.loc[0, 'A(t)'] = 0
for i in range(1, len(df_true1)):
	df_true1.loc[i, 'A(t)'] = df_true1.loc[i-1, 'N(t)'] + df_true1.loc[i-1, 'A(t)']

df_noise1.loc[0, 'A(t)'] = 0
for i in range(1, len(df_noise1)):
	df_noise1.loc[i, 'A(t)'] = df_noise1.loc[i-1, 'N(t)'] + df_noise1.loc[i-1, 'A(t)']

df_true1['A(t)^2'] = df_true1['A(t)'] ** 2
df_noise1['A(t)^2'] = df_noise1['A(t)'] ** 2

# initialize regression model for true values
lin_reg = LinearRegression()
# define X variables, feature names, and Y
X = df_true1.drop(columns=['t', 'N(t)'])
# features = X.columns
Y = df_true1['N(t)']
# fit model
lin_reg.fit(X, Y)

# assign coeff values
a = lin_reg.intercept_
b, c = lin_reg.coef_[0], lin_reg.coef_[1]
# solve for p, q, M, and N(30)
p = (np.sqrt(b**2 - 4*a*c) - b) / 2
q = (np.sqrt(b**2 - 4*a*c) + b) / 2
M = -q / c
# display values
print(f'True c-value for low-error case: c = {c:.5f}')

# extrapolate
df_true1 = extrap_discrete_bass(df_true1, p, q, M, 30)

# initialize regression model for noisy values
lin_reg = LinearRegression()
# define X variables, feature names, and Y
X = df_noise1.drop(columns=['t', 'N(t)'])
# features = X.columns
Y = df_noise1['N(t)']
# fit model
lin_reg.fit(X, Y)

# assign coefficient values
a = lin_reg.intercept_
b, c = lin_reg.coef_[0], lin_reg.coef_[1]
# solve for p, q, M, and N(30)
p = (np.sqrt(b**2 - 4*a*c) - b) / 2
q = (np.sqrt(b**2 - 4*a*c) + b) / 2
M = -q / c
print(f'Noisy c-value for low-error case: c = {c:.5f}')

# extrapolate
df_noise1 = extrap_discrete_bass(df_noise1, p, q, M, 30)

# plot together
plt.plot(df_true1['t'], df_true1['N(t)'], 'o', c='green')
plt.plot(df_noise1['t'], df_noise1['N(t)'], 'o', c='red')
plt.text(1, 1, s=f'Noisy values: p = {p:5f}\nq = {q:5f}\nM = {M:5f}')
plt.legend(['True', 'Noise'])
plt.title('Noisy Data: Results in Low Error')
plt.xlabel('t')
plt.ylabel('N(t)')
plt.show()



# HIGH-ERROR CASE
# populating each dataframe with values for A(t) and A(t)^^2
df_true2.loc[0, 'A(t)'] = 0
for i in range(1, len(df_true2)):
	df_true2.loc[i, 'A(t)'] = df_true2.loc[i-1, 'N(t)'] + df_true2.loc[i-1, 'A(t)']

df_noise2.loc[0, 'A(t)'] = 0
for i in range(1, len(df_noise2)):
	df_noise2.loc[i, 'A(t)'] = df_noise2.loc[i-1, 'N(t)'] + df_noise2.loc[i-1, 'A(t)']

df_true2['A(t)^2'] = df_true2['A(t)'] ** 2
df_noise2['A(t)^2'] = df_noise2['A(t)'] ** 2

# initialize regression model for true values
lin_reg = LinearRegression()
# define X variables, feature names, and Y
X = df_true2.drop(columns=['t', 'N(t)'])
# features = X.columns
Y = df_true2['N(t)']
# fit model
lin_reg.fit(X, Y)

# assign coefficient values
a = lin_reg.intercept_
b, c = lin_reg.coef_[0], lin_reg.coef_[1]
# solve for p, q, M, and N(30)
p = (np.sqrt(b**2 - 4*a*c) - b) / 2
q = (np.sqrt(b**2 - 4*a*c) + b) / 2
M = -q / c
print(f'True c-value for high-error case: c = {c:.5f}')

# extrapolate
df_true2 = extrap_discrete_bass(df_true2, p, q, M, 30)

# initialize regression model for noisy values
lin_reg = LinearRegression()
# define X variables, feature names, and Y
X = df_noise2.drop(columns=['t', 'N(t)'])
# features = X.columns
Y = df_noise2['N(t)']
# fit model
lin_reg.fit(X, Y)

# assign coefficient values
a = lin_reg.intercept_
b, c = lin_reg.coef_[0], lin_reg.coef_[1]
# solve for p, q, M, and N(30)
p = (np.sqrt(b**2 - 4*a*c) - b) / 2
q = (np.sqrt(b**2 - 4*a*c) + b) / 2
M = -q / c
print(f'Noisy c-value for high-error case: c = {c:.5f}')

# extrapolate
df_noise2 = extrap_discrete_bass(df_noise2, p, q, M, 30)

# plot together
plt.plot(df_true2['t'], df_true2['N(t)'], 'o', c='green')
plt.plot(df_noise2['t'], df_noise2['N(t)'], 'o', c='red')
plt.text(1, 1, s=f'Noisy values: p = {p:5f}\nq = {q:5f}\nM = {M:5f}')
plt.legend(['True', 'Noise'])
plt.title('Noisy Data: Results in High Error')
plt.xlabel('t')
plt.ylabel('N(t)')
plt.show()

print('The main difference between the low- and high-error cases is that the low-error case has a substantially better estimate of c.')
print('See writeup for more details.')
print()



print('QUESTION 4---------------------------------------------')
df_4 = pd.read_excel('adoptionseries2_with_noise.xlsx')
df_4.loc[0, 'A(t)'] = 0
for i in range(1, len(df_4)):
	df_4.loc[i, 'A(t)'] = df_4.loc[i-1, 'N(t)'] + df_4.loc[i-1, 'A(t)']

p_vals = np.linspace(0.003, 0.3, 10)
q_vals = np.linspace(0.006, 0.6, 10)
M = 100

# storing values to plot
p_store = []
q_store = []
N_discrete_store = []
N_cont_store = []

for p in p_vals:
	for q in q_vals:
		df_4_discrete = extrap_discrete_bass(df_4, p, q, M, 30)
		N_discrete = df_4_discrete.loc[29, "N(t)"]
		df_4_cont = extrap_cont_bass(df_4, p, q, M, 30)
		N_cont = df_4_cont.loc[29, "N(t)"]

		p_store.append(p)
		q_store.append(q)
		N_discrete_store.append(N_discrete)
		N_cont_store.append(N_cont)

		print(f'p = {p:.5f}, q = {q:.5f}')
		print(f'Discrete N(30)' = {N_discrete:.2})
		print(f'Continuous N(30)' = {N_cont:.2})


df_4_pq = pd.DataFrame({'p': p_store, 'q': q_store,
	'N(t) (discr)': N_discrete_store,
	'N(t) (cont)': N_cont_store})

# plotting trends of p and q vs. both N(t) versions
plt.plot(df_4_pq['p'], df_4_pq['N(t) (discr)'])
plt.xlabel('p')
plt.ylabel('N(t) (discr)')
plt.title('p vs. Discrete N(t)')
plt.show()


plt.plot(df_4_pq['p'], df_4_pq['N(t) (cont)'])
plt.xlabel('p')
plt.ylabel('N(t) (cont)')
plt.title('p vs. Cont. N(t)')
plt.show()

plt.plot(df_4_pq['q'], df_4_pq['N(t) (discr)'])
plt.xlabel('q')
plt.ylabel('N(t) (discr)')
plt.title('q vs. Discrete N(t)')
plt.show()


plt.plot(df_4_pq['q'], df_4_pq['N(t) (cont)'])
plt.xlabel('q')
plt.ylabel('N(t) (cont)')
plt.title('q vs. Cont. N(t)')
plt.show()

# plotting discrete vs. continuous

# subsetting dataframes based on relationship between p and q
df_4_pq = df_4_pq.loc[df_4_pq['p'] < df_4_pq['q']]
# df_4_pq = df_4_pq.loc[df_4_pq['p'] > df_4_pq['q']]
X = df_4_pq['N(t) (discr)']
Y = df_4_pq['N(t) (cont)']
plt.plot(X, Y, 'o', c='blue')
# plot Y = X
coords = [i for i in range(4)]
plt.plot(coords, coords, c='r')
plt.xlabel('N(t) (discr)')
plt.ylabel('N(t) (cont)')
plt.title('Discr. vs. Cont., p < q')
plt.show()

print()

print('QUESTION 5---------------------------------------------')
print('See writeup.')
print()