import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt

# Functions defined to find the values of exp(x) and cos(cos(x))
def f1(x):
	return np.exp(x)

def f2(x):
	return np.cos(np.cos(x))

# Function to find the zeroth coefficient of Fourier Series
def ao():
	ao1 = integrate.quad(lambda x: f1(x), 0, 2*np.pi)
	ao2 = integrate.quad(lambda x: f2(x), 0, 2*np.pi)
	return ao1, ao2

# Fuction to find cosine coefficients of Fourier Series
def an(n):
	an1 = integrate.quad(lambda x: f1(x)*np.cos(n*x), 0, 2*np.pi)
	an2 = integrate.quad(lambda x: f2(x)*np.cos(n*x), 0, 2*np.pi)
	return an1, an2

# Function to find sine coefficients of Fourier Series
def bn(n):
	bn1 = integrate.quad(lambda x: f1(x)*np.sin(n*x), 0, 2*np.pi)
	bn2 = integrate.quad(lambda x: f2(x)*np.sin(n*x), 0, 2*np.pi)
	return bn1, bn2


'''
Q1
Plotting the functions in given range
'''
x = np.arange(-2*np.pi, 4*np.pi, 0.01)
y1 = f1(x)
y2 = f2(x)

# Plotting exp(x) in Figure 1
plt.figure(1)
plt.yscale('log')
plt.title('exp(x)')
plt.grid()
plt.xlabel(r'x $\longrightarrow$')
plt.ylabel(r'y (log scale) $\longrightarrow$')
plt.plot(x, y1)
plt.show()

# Plotting cos(cos(x)) in Figure 2
plt.figure(2)
plt.xlabel(r'x $\longrightarrow$')
plt.ylabel(r'y $\longrightarrow$')
plt.grid()
plt.title('cos(cos(x))')
plt.plot(x, y2)
plt.show()

'''
Q2
51 coefficients
'''
# Array to store values from 1 to 25
coeffs = np.arange(1,26,1)
# Array to hold all the coefficcients of f1
f1_co = np.zeros(51)
# Array to hold all the coefficients of f2
f2_co = np.zeros(51)
# Array to hold all the an coefficients of f1
an1 = np.zeros(25)
# Array to hold all the an coefficients of f2
an2 = np.zeros(25)
# Array to hold all the bn coefficients of f1
bn1 = np.zeros(25)
# Array to hold all the bn coefficients of f2
bn2 = np.zeros(25)
# ao() returns the a0 of f1 and f2
ao1_, ao2_ = ao()
# Exteacting the integrated value for both f1 and f2
ao1 = ao1_[0]
ao2 = ao2_[0]
f1_co[0] = ao1
f2_co[0] = ao2

# Feeding values to the coefficient finding functions one by one from 1 to 25
for n in coeffs:
	# Finding an for both f1 and f2
	an1_, an2_ = an(n)
	# Finding bn for both f1 and f2
	bn1_, bn2_ = bn(n)
	an1[n-1] = an1_[0]
	an2[n-1] = an2_[0]
	bn1[n-1] = bn1_[0]
	bn2[n-1] = bn2_[0]
	# Assigning values to appropriate indices of the coefficient vector
	f1_co[2*n-1] = an1_[0]
	f2_co[2*n-1] = an2_[0]
	f1_co[2*n] = bn1_[0]
	f2_co[2*n] = bn2_[0]


'''
Q3
Plots of magnitude of coefficients versus n
'''
# First plotting with a semilog scale
# exp(x)
plt.figure(3)
plt.grid()
plt.yscale('log')
plt.title('Coeffiecients of exp(x) on semilog scale')
plt.xlabel(r'n $\longrightarrow$')
plt.ylabel(r'Coefficients $\longrightarrow$')
plt.plot(coeffs, abs(an1), 'ro', label = r'$a_n$') 
plt.plot(0, ao1, 'ro')
plt.plot(coeffs, abs(bn1), 'go', label = r'$b_n$')
plt.legend()
plt.show()
# cos(cos(x))
plt.figure(5)
plt.grid()
plt.yscale('log')
plt.title('Coefficients of cos(cos(x)) on semilog scale')
plt.xlabel(r'n $\longrightarrow$')
plt.ylabel(r'Coefficients $\longrightarrow$')
plt.plot(coeffs, abs(an2), 'ro', label = r'$a_n$') 
plt.plot(0, ao1, 'ro')
plt.plot(coeffs, abs(bn2), 'go', label = r'$b_n$')
plt.legend()
plt.show()
# Now plotting with a loglog scale
# exp(x)
plt.figure(4)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.title('Coeffiecients of exp(x) on loglog scale')
plt.xlabel(r'n $\longrightarrow$')
plt.ylabel(r'Coefficients $\longrightarrow$')
plt.plot(coeffs, abs(an1), 'ro', label = r'$a_n$') 
plt.plot(0, ao1, 'ro')
plt.plot(coeffs, abs(bn1), 'go', label = r'$b_n$')
plt.legend()
plt.show()
# cos(cos(x))
plt.figure(6)
plt.grid()
plt.xscale('log')
plt.yscale('log')
plt.title('Coefficients of cos(cos(x)) on loglog scale')
plt.xlabel(r'n $\longrightarrow$')
plt.ylabel(r'Coefficients $\longrightarrow$')
plt.plot(coeffs, abs(an2), 'ro', label = r'$a_n$') 
plt.plot(0, ao1, 'ro')
plt.plot(coeffs, abs(bn2), 'go', label = r'$b_n$')
plt.legend()
plt.show()

'''
Q4, Q5
Least Squares Approximation
'''
# Generating 401 points from 0 to 2pi
x = np.linspace(0, 2*np.pi, 401)
x = x[:-1]
A = np.zeros((400, 51))
b1 = f1(x)
b2 = f2(x)
A[:,0] = 1
# Generating the A matrix
for k in range(1, 26):
	A[:,2*k-1] = np.cos(k*x)
	A[:, 2*k] = np.sin(k*x)

# Applying the least squares method to find the 51 coefficients
ls_co1 = np.linalg.lstsq(A, b1, rcond=None)[0]
ls_co2 = np.linalg.lstsq(A, b2, rcond=None)[0]

# Declaring coeffs from 0 to 50 for plotting
coeffs = np.arange(0, 51, 1)

# exp(x) coefficients
plt.figure(7)
plt.grid()
plt.yscale('log')
plt.title('Coefficients obtained by lstsq vs Integration for exp(x)')
plt.xlabel(r'n $\longrightarrow$')
plt.ylabel(r'Coefficients $\longrightarrow$')
plt.plot(coeffs, abs(f1_co), 'ro', label='Integration coefficients') 
plt.plot(coeffs, abs(ls_co1), 'go', label='Least squares coefficients') 
plt.legend()
plt.show()

# cos(cos(x)) coefficients
plt.figure(8)
plt.grid()
plt.yscale('log')
plt.title('Coefficients obtained by lstsq vs Integration for cos(cos(x))')
plt.xlabel(r'n $\longrightarrow$')
plt.ylabel(r'Coefficients $\longrightarrow$')
plt.plot(coeffs, abs(f2_co), 'ro', label='Integration coefficients') 
plt.plot(coeffs, abs(ls_co2), 'go', label='Least squares coefficients') 
plt.legend()
plt.show()

'''
Q6
Error in coefficients obtained least squares method
'''
# Finding absolute error for each of the functions
err1 = np.abs(f1_co-ls_co1)
print('Largest deviation in coefficients of exp(x): %f' %(np.amax(err1)))
err2 = np.abs(f2_co-ls_co2)
print('Largest deviation in coefficients of cos(cos(x)): %f' %(np.amax(err2)))

'''
Q7
Plotting Ac against the actual function
'''
# Finding the value of the functions through the coefficients found by least 
# squares approximation
f1_ls = A.dot(ls_co1)
f2_ls = A.dot(ls_co2)

# Plotting actual exp(x) and approximate value
plt.figure(1)
plt.grid()
plt.yscale('log')
plt.xlabel(r'x $\longrightarrow$')
plt.ylabel(r'$exp(x)$ $\longrightarrow$')
plt.title('Least Squares approximation vs Actual Function')
plt.plot(x, f1_ls, 'go', label='Least Squares approximation')
plt.plot(x, np.exp(x), 'r', label='Actual Plot')
plt.legend()
plt.show()

# Plotting actual  cos(cos(x)) and approximate value
plt.figure(2)
plt.grid()
plt.xlabel(r'x $\longrightarrow$')
plt.ylabel(r'$cos(cos(x))$ $\longrightarrow$')
plt.title('Least Squares approximation vs Actual Function')
plt.plot(x, f2_ls, 'go', label='Least Squares approximation')
plt.plot(x, np.cos(np.cos(x)), 'r', label='Actual Plot')
plt.legend()
plt.show()




















