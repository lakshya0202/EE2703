from sympy import *
import scipy.signal as sp
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
import numpy as np
import warnings

'''
Q1
Simulating the circuit shown above
'''
def lowpass(R1, R2, C1, C2, G, Vi):
	s = symbols('s') 
	A = Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0], [0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]]) 
	b = Matrix([0,0,0,Vi/R1])
	V = A.inv()*b
	return (A, b, V)

# Declaring the symbol s
s = symbols('s')
# Calling the lowpass function by passing the appropriate values
A,b,V = lowpass(10000, 10000, 1e-9, 1e-9, 1.586, 1) 
# Creating a logspace values array from 1 to 10^8
w = np.logspace(0,8,801) 
# Creating variable ss to hold the imaginary jw variable
ss = 1j*w 
# Plotting the function available at the third position of the V array
Vo = V[3]
# Using lambdify to express V as a function
hf = lambdify(s, Vo, 'numpy') 
# Giving the s symbol the value jw
v = hf(ss) 
# Plotting the output figure
plt.figure()
plt.title("Plot of transfer function of Vo in LPF implementation")
plt.xlabel(r'$\omega$')
plt.ylabel('Magnitude')
plt.loglog(w, abs(v), lw=2)
plt.grid(True)
plt.show()
# Now obtaining the step response of the above circuit using lsim
t = np.linspace(0, 10, 101)
# Using simplify to make the function easier to read
Vo = Vo.simplify()
# Using Poly from the sympy libraray to extract the coefficients of numerator and denominator
num = Poly(Vo.as_numer_denom()[0], s).all_coeffs()
den = Poly(Vo.as_numer_denom()[1], s).all_coeffs()
# Converting the extracted coefficients to float type
num = np.array(num, dtype = float)
den = np.array(den, dtype = float)
# using lti to generate the transfer function
H = sp.lti(num, den)
vi = np.ones(101)
# Filtering any unnecessary warning which may be thrown
warnings.filterwarnings("ignore")
# Using lsim to generate the time dependent plot
t, y, svec = sp.lsim(H, vi, t)
# Plotting the final results
plt.figure()
plt.title(r'Output unit step input function')
plt.xlabel(r't $\longrightarrow$')
plt.ylabel(r'$v_o(t)$ $\longrightarrow$')
plt.plot(t, y)
plt.grid(True)
plt.show()


'''
Q2, Q3, Q4, Q5
HPF Circuit
'''
def highpass(R1, R3, C1, C2, G, Vi):
	s = symbols('s') 
	A = Matrix([[0,0,1,-1/G],[-1/(1+(1/(s*C2*R3))),1,0,0], [0,-G,G,1],[-s*C1-s*C2-(1/R1),s*C2,0,1/R1]]) 
	b = Matrix([0,0,0,Vi*s*C1])
	V = A.inv()*b
	return (A, b, V)

'''
Plotting the transfer function of output for a constant input of 1
'''
# Declaring the symbol s
s, t = symbols('s, t')
# Calling the highpass function by passing the appropriate values
# Finding the transfer function for constant input of 1
A,b,V = highpass(10000, 10000, 1e-9, 1e-9, 1.586, 1) 
# Creating a logspace values array from 1 to 10^8
w = np.logspace(0,9,901) 
# Creating variable ss to hold the imaginary jw variable
ss = 1j*w 
# Plotting the function available at the third position of the V array
Vo = V[3]
# Using lambdify to express V as a function
hf = lambdify(s, Vo, 'numpy') 
# Giving the s symbol the value jw
v = hf(ss) 
# Plotting the output figure
plt.figure()
plt.title("Plot of transfer function of Vo for Vi = 1")
plt.xlabel(r'$\omega$')
plt.ylabel('Magnitude')
plt.loglog(w, abs(v), lw=2)
plt.grid(True)
plt.show()


'''
Plotting the transfer function of output for a cos + sin combination
'''
# Calling the highpass function by passing the appropriate values
# Finding the transfer function for given input of cos and sin
f = cos(2e6*np.pi*t) + sin(2000*np.pi*t)
vi_L = laplace_transform(f, t, s, noconds=True)
A,b,V_ = highpass(10000, 10000, 1e-9, 1e-9, 1.586, vi_L) 
# Plotting the function available at the third position of the V array
Vo_ = V_[3]
# Using lambdify to express V as a function
hf_ = lambdify(s, Vo_, 'numpy') 
# Giving the s symbol the value jw
v_ = hf_(ss) 
# Plotting the output figure
plt.figure()
plt.title("Plot of transfer function of Vo for sin + cos input")
plt.xlabel(r'$\omega$')
plt.ylabel('Magnitude')
plt.loglog(w, abs(v_), lw=2)
plt.grid(True)
plt.show()


'''
Plotting the transfer function of output for a decaying sinusoid
'''
f = cos(2000*np.pi*t)*exp(-0.5*t)
vi_L = laplace_transform(f, t, s, noconds=True)
A,b,V_ = highpass(10000, 10000, 1e-9, 1e-9, 1.586, vi_L) 
# Plotting the function available at the third position of the V array
Vo_ = V_[3]
# Using lambdify to express V as a function
hf_ = lambdify(s, Vo_, 'numpy') 
# Giving the s symbol the value jw
v_ = hf_(ss) 
# Plotting the output figure
plt.figure()
plt.title("Plot of transfer function of Vo for decaying sinusiod input")
plt.xlabel(r'$\omega$')
plt.ylabel('Magnitude')
plt.loglog(w, abs(v_), lw=2)
plt.grid(True)
plt.show()


'''
Plotting the transfer function of output for a unit step
'''
A,b,V_ = highpass(10000, 10000, 1e-9, 1e-9, 1.586, 1/s) 
# Plotting the function available at the third position of the V array
Vo_ = V_[3]
# Using lambdify to express V as a function
hf_ = lambdify(s, Vo_, 'numpy') 
# Giving the s symbol the value jw
v_ = hf_(ss) 
# Plotting the output figure
plt.figure()
plt.title("Plot of transfer function of Vo for unit step input")
plt.xlabel(r'$\omega$')
plt.ylabel('Magnitude')
plt.loglog(w, abs(v_), lw=2)
plt.grid(True)
plt.show()


'''
Plotting the time varying response of each of the inputs
First the sin and cos combination
'''
# Using simplify to make the function easier to read
Vo = Vo.simplify()
# Using Poly from the sympy libraray to extract the coefficients of numerator and denominator
num = Poly(Vo.as_numer_denom()[0], s).all_coeffs()
den = Poly(Vo.as_numer_denom()[1], s).all_coeffs()
# Converting the extracted coefficients to float type
num = np.array(num, dtype = float)
den = np.array(den, dtype = float)
# Creating the time array
time = np.linspace(0, 0.005, 100000)


# Finding the laplace transforms of the input function
f = cos(2e6*np.pi*t) + sin(2000*np.pi*t)
vi_L = laplace_transform(f, t, s, noconds=True)
vi_L = vi_L.simplify()
# Writing the input as a polynomial
num1 = Poly(vi_L.as_numer_denom()[0], s).all_coeffs()
den1 = Poly(vi_L.as_numer_denom()[1], s).all_coeffs()
# Converting coefficients to float data type
num1 = np.poly1d(np.array(num1, dtype = float))
den1 = np.poly1d(np.array(den1, dtype = float))
# Multiplying the numerators and denominators
numf = np.polymul(num, num1)
denf = np.polymul(den, den1)
# using lti to generate the transfer function
H1 = sp.lti(numf, denf)
warnings.filterwarnings("ignore")
# Using lsim to generate the time dependent plot
time, y, svec = sp.lsim(H1, np.ones(time.shape), time)
# Plotting the final results
plt.figure()
plt.title(r'Output for $\cos(2*10^6 \pi t)+sin(2000 \pi t)$')
plt.xlabel(r't $\longrightarrow$')
plt.ylabel(r'$v_o(t)$ $\longrightarrow$')
plt.plot(time, y)
plt.grid(True)
plt.show()


'''
Now the decaying sinusoid
'''
f = cos(2000*np.pi*t)*exp(-0.5*t)
vi_L = laplace_transform(f, t, s, noconds=True)
vi_L = vi_L.simplify()
# Writing the input as a polynomial
num2 = Poly(vi_L.as_numer_denom()[0], s).all_coeffs()
den2 = Poly(vi_L.as_numer_denom()[1], s).all_coeffs()
# Converting coefficients to float data type
num2 = np.array(num2, dtype = float)
den2 = np.array(den2, dtype = float)
# Multiplying the numerators and denominators
numf = np.polymul(num, num2)
denf = np.polymul(den, den2)
# using lti to generate the transfer function
H2 = sp.lti(numf, denf)
warnings.filterwarnings("ignore")
# Using lsim to generate the time dependent plot
time, y, svec = sp.lsim(H2, np.ones(time.shape), time)
# Plotting the final results
plt.figure()
plt.title(r'Output for exponentially decaying sinusoid')
plt.xlabel(r't $\longrightarrow$')
plt.ylabel(r'$v_o(t)$ $\longrightarrow$')
plt.plot(time, y)
plt.grid(True)
plt.show()


'''
Now the simple unit step
'''
vi_L = 1/s
vi_L = vi_L.simplify()
# Writing the input as a polynomial
num3 = Poly(vi_L.as_numer_denom()[0], s).all_coeffs()
den3 = Poly(vi_L.as_numer_denom()[1], s).all_coeffs()
# Converting coefficients to float data type
num3 = np.array(num3, dtype = float)
den3 = np.array(den3, dtype = float)
# Multiplying the numerators and denominators
numf = np.polymul(num, num3)
denf = np.polymul(den, den3)
# using lti to generate the transfer function
H3 = sp.lti(numf, denf)
warnings.filterwarnings("ignore")
# Using lsim to generate the time dependent plot
time, y, svec = sp.lsim(H3, np.ones(time.shape), time)
# Plotting the final results
plt.figure()
plt.title(r'Output for unit step function')
plt.xlabel(r't $\longrightarrow$')
plt.ylabel(r'$v_o(t)$ $\longrightarrow$')
plt.plot(time, y)
plt.grid(True)
plt.show()









