import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

'''
fitting.dat file is assumed to be present in the current directory
as a result of running the code sir had given previously
'''

'''
Function to define g(t)
'''
def g(t, A, B):
	return A*scipy.special.jv(2, t)+B*t

'''
Q2
'''
data_points = np.loadtxt('fitting.dat')

# f_t is the data points obtained from fitting.dat after removing the time stamps
f_t = np.zeros((np.shape(data_points)[0], np.shape(data_points)[1]-1))
f_t = data_points[:,1:]
# x stores all the time stamps
t = data_points[:,0]

# y stores all the jv values in a 101 X 9 matrix
y = scipy.special.jv(2,t)*-1.05
y = y.reshape((101,1))
x = t.reshape((101,1))*0.105
# t_matrix is the matrix formed by concatenating all the column vectors
t_matrix = np.hstack((x,x,x,x,x,x,x,x,x))
# J2_matrix is the matris formed by concatenating all the jv values in column vectors
J2_matrix = np.hstack((y,y,y,y,y,y,y,y,y))
# Generating the range of sigma values using logspace
sigma = np.logspace(-1, -3, 9)
sig = np.round(np.logspace(-1,-3, 9), 3)
sigma = sigma.reshape((9,1))

# Matrix to hold various n(t) values
n_t = np.zeros(np.shape(f_t))
# Matrix addition to find the noise matrix
n_t = f_t + J2_matrix +t_matrix

'''
Q3, Q4
Plotting the noise of all the functions as a function of time dependent
on the sigma at that point
'''
A = 1.05
B = -0.105
g_t = g(t, A, B)
plt.figure(0)
plt.plot(t, f_t[:,0], t, f_t[:,1], t, f_t[:,2], t, f_t[:,3], t, f_t[:,4], t,f_t[:,5], t, f_t[:,6], t, f_t[:,7], t, f_t[:,8], t, g_t,'black')
plt.legend([r"$\sigma_1$ = {}".format(sig[0]),r"$\sigma_2$ = {}".format(sig[1]),r"$\sigma_3$ = {}".format(sig[2]),r"$\sigma_4$ = {}".format(sig[3]),r"$\sigma_5$ = {}".format(sig[4]),r"$\sigma_6$ = {}".format(sig[5]),r"$\sigma_7$ = {}".format(sig[6]),r"$\sigma_8$ = {}".format(sig[7]),r"$\sigma_9$ = {}".format(sig[8]), r"True Value"])
plt.xlabel(r'$Time \longrightarrow$')
plt.ylabel(r'$Noise \longrightarrow$')
plt.title('Noise as a function of time')
plt.show()


'''
Q5
Error bars for the first column of data
'''
stdev = np.std(f_t, axis=0)[0]
col1=np.zeros((101,1))
# extracting the first column of f_t
col1=f_t[:,0]
plt.figure(1)
plt.errorbar(t[::5], col1[::5], yerr=stdev, fmt = 'ro')
plt.plot(t, g_t, 'black')
plt.legend(["g(t, A, B)", "Error f(t)"])
plt.title(r'Error bar for $\sigma_1$')
plt.xlabel(r'$t \longrightarrow$')
plt.show()

'''
Q6
'''
j2 = scipy.special.jv(2,t)
t = t.reshape((101,1))
j2 = j2.reshape((101,1))
J_tmatrix = np.hstack([j2, t])
AB = np.array([A,B])
g_matrix = np.matmul(J_tmatrix, AB)
# checking if the arrays are equal using a numpy package
if np.array_equal(g_matrix, g_t):
	print("The arrays are equal")

'''
Q7
Mean Squared Error
'''
A_array = np.arange(0, 2.1, 0.1)
B_array = np.arange(-0.2, 0.01, 0.01)

Eij = np.zeros((21,21))

for i in range(21):
	for j in range(21):
		gk = g(t, A_array[i], B_array[j])
		col1 = col1.reshape(np.shape(t))
		e = np.sum(np.square(col1-gk))/101
		Eij[i][j] = e

'''
Q8
Contour plot
'''
plt.figure(3)
cs = plt.contour(A_array, B_array, Eij, levels = np.linspace(0,0.3,10))
plt.clabel(cs, inline=True)
plt.plot(A, B, 'ro')
plt.annotate('Exact Location', (A, B))
plt.title(r'Contour plot for $\epsilon_{ij}$')
plt.xlabel(r'$A \longrightarrow$')
plt.ylabel(r'$B \longrightarrow$')
plt.show()

'''
Q9
Estimate of A and B
'''
ans = lstsq(J_tmatrix, g_t)
print("A=%f" %ans[0][0])
print("B=%f" %ans[0][1])
# This obviously will print the exact values

'''
Q10
Repeating for columns 1 to 9 with different noise values
'''
A_error = np.zeros(9)
B_error = np.zeros(9)
A_rand = np.zeros(9)
B_rand = np.zeros(9)
j2 = scipy.special.jv(2,t)
j2 = j2.reshape((101,1))
J_tmatrix = np.hstack([j2, t])
# Loop to store all values and errors on A and B in numpy arrays
for i in range(0,9):
	col = f_t[:,i]
	ans = lstsq(J_tmatrix, col)
	A_rand[i] = ans[0][0]
	B_rand[i] = ans[0][1]
	# Calculating absolute value between approximate and true values of A and B
	A_error[i] = abs(ans[0][0] - A)
	B_error[i] = abs(ans[0][1] - B)

plt.figure(4)
plt.plot(sigma, A_error, 'ro--', sigma, B_error, 'go--')
plt.legend(['A', 'B'])
plt.title('Error in A and B')
plt.xlabel(r'Noise Standard Deviation $\longrightarrow$')
plt.ylabel(r'$Error \longrightarrow$')
plt.show()

'''
Q11
Replotting the same curve in loglog
'''
plt.figure(5)
plt.xscale('log')
plt.yscale('log')
sigma = np.logspace(-1, -3, 9)
plt.errorbar(sigma, A_error, yerr = A_error, fmt='ro')
plt.errorbar(sigma, B_error, yerr = B_error, fmt='go')
plt.legend(['Aerr', 'Berr'])
plt.title("loglog plot for the error in A and B")
plt.xlabel(r'Noise Standard Deviation $\longrightarrow$')
plt.ylabel(r'$Error \longrightarrow$')
plt.show()





















