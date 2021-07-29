import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt

'''
Q1
'''
num = np.poly1d([1,0.5])
den = np.polymul([1,0,2.25],[1,1,2.5])
H = [num, den]
t, x = sp.impulse(H, None, np.linspace(0,50,501))
plt.figure()
plt.plot(t,x)
plt.grid(True)
plt.title(r"Time response of spring $\alpha$ $=$ 0.5")
plt.xlabel(r"t $\longrightarrow$")
plt.ylabel(r"f $\longrightarrow$")
plt.show()

'''
Q2
'''
# Changing the values of the polynomial appropriately
# To reflect the changes in the f function
num = np.poly1d([1,0.05])
den = np.polymul([1,0,2.25],[1,0.1,2.2525])
H = [num, den]
t, x = sp.impulse(H, None, np.linspace(0,50,501))
plt.figure()
plt.plot(t,x)
plt.grid(True)
plt.title(r"Time response of spring $\alpha$ $=$ 0.05")
plt.xlabel(r"t $\longrightarrow$")
plt.ylabel(r"f $\longrightarrow$")
plt.show()

'''
Q3
'''
# Since X(s) = F(s)/(s^2 + 2.25), H(s) = 1/(s^2 + 2.25)
num = np.poly1d([1])
den = np.poly1d([1,0,2.25])
H = [num, den]
freq = np.arange(1.4, 1.65, 0.05)
t = np.linspace(0, 50, 501)
plt.figure()
plt.grid(True)
plt.title(r"Simulation using $lsim$")
plt.xlabel(r"t $\longrightarrow$")
plt.ylabel(r"y $\longrightarrow$")
for i in freq:
	u = np.cos(i*t)*np.exp(-0.05*t)
	t, y, svec = sp.lsim(H, u, t)
	plt.plot(t, y)
plt.legend([r"$\omega$ $=$ {}".format(freq[0]), r"$\omega$ $=$ {}".format(freq[1]), r"$\omega$ $=$ {}".format(freq[2]), r"$\omega$ $=$ {}".format(freq[3]), r"$\omega$ $=$ {}".format(freq[4])])
plt.show()

'''
Q4
Coupled spring problem
'''
# Solving for x vs time in laplace domain
numX = np.poly1d([1,0,2])
denX = np.poly1d([1,0,3,0])
X = [numX, denX]
t, x = sp.impulse(X, None, np.linspace(0,20,501))
# Solving for y vs time in laplace domain
numY = np.poly1d([2])
denY = np.poly1d([1,0,3,0])
Y = [numY, denY]
t, y = sp.impulse(Y, None, np.linspace(0,20,501))
# Plotting the figures accordingly
plt.figure()
plt.grid(True)
plt.title(r'Coupled spring problem')
plt.xlabel(r't $\longrightarrow$')
plt.ylabel(r'Displacement $\longrightarrow$')
plt.plot(t, x)
plt.plot(t, y)
plt.legend(['x', 'y'])
plt.show()

'''
Q5
Steady State Transfer function of two port network
'''
# The laplace output function 
# vo(s)/vi(s) = 1/(s^2LC + sRC + 1)
L = 1e-6
C = 1e-6
R = 100
H = sp.lti([1],[L*C, R*C, 1])
w, S, phi = sp.bode(H)
plt.figure()
plt.subplot(2,1,1)
plt.title('Magnitude Plot')
plt.xlabel(r'$\omega$ $\longrightarrow$')
plt.ylabel('|H(s)|')
plt.semilogx(w, S)
plt.subplot(2,1,2)
plt.title('Phase Plot')
plt.xlabel(r'$\omega$ $\longrightarrow$')
plt.ylabel(r'$\angle$H(s)')
plt.semilogx(w, phi)
plt.show()

'''
Q6
Applying the same transfer function as above but with different input
'''
t = np.linspace(0, 0.02, 1000000)
vi = np.cos((10**3)*t) - np.cos((10**6)*t)
t, y, svec = sp.lsim(H, vi, t)
plt.figure()
plt.title(r'Output with different input function')
plt.xlabel(r't $\longrightarrow$')
plt.ylabel(r'$v_o(t)$ $\longrightarrow$')
plt.plot(t, y, 'g')
plt.grid(True)
plt.show()















