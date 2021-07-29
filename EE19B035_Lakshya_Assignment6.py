from sys import argv, exit
import numpy as np
import matplotlib.pyplot as plt
import random

n=100
M=5
nk=500
u0=5
p=0.25
Msig=2

if len(argv)==1:
	print("Default values have been assigned")

elif len(argv)==2:
	print("n has been assigned %s" %argv[1])
	n=int(argv[1])

elif len(argv)==3:
	print("n has been assigned %s" %argv[1])
	print("M has been assigned as %s" %argv[2])
	n=int(argv[1])
	M=int(argv[2])
	
elif len(argv)==4:
	print("n has been assigned %s" %argv[1])
	print("M has been assigned as %s" %argv[2])
	print("nk has been assigned %s" %argv[3])
	n=int(argv[1])
	M=int(argv[2])
	nk=int(argv[3])

elif len(argv)==5:
	print("n has been assigned %s" %argv[1])
	print("M has been assigned as %s" %argv[2])
	print("nk has been assigned %s" %argv[3])
	print("u0 has been assigned %s" %argv[4])
	n=int(argv[1])
	M=int(argv[2])
	nk=int(argv[3])
	u0=int(argv[4])

elif len(argv)==6:
	print("n has been assigned %s" %argv[1])
	print("M has been assigned as %s" %argv[2])
	print("nk has been assigned %s" %argv[3])
	print("u0 has been assigned %s" %argv[4])
	print("p has been assigned %s" %argv[5])
	n=int(argv[1])
	M=int(argv[2])
	nk=int(argv[3])
	u0=int(argv[4])
	p=float(argv[5])

elif len(argv)==7:
	print("n has been assigned %s" %argv[1])
	print("M has been assigned as %s" %argv[2])
	print("nk has been assigned %s" %argv[3])
	print("u0 has been assigned %s" %argv[4])
	print("p has been assigned %s" %argv[5])
	print("Msig has been assigned %s" %argv[6])
	n=int(argv[1])
	M=int(argv[2])
	nk=int(argv[3])
	u0=int(argv[4])
	p=float(argv[5])
	Msig=float(argv[6])

# Electron position
xx = np.zeros(n*M)
# Electron velocity
u = np.zeros(n*M)
# Displacement in current turn
dx = np.zeros(n*M)

# Intensity of Light emitted
I = []
# Electron position
X =[]
# Electron velocity
V =[]
# All positions where electrons have been generated
ii = np.where(xx>0)[0]

for i in range(nk):
	# Changing position due to acceleration
	dx[ii] = u[ii]+0.5
	# Changing position along x axis of accelerated electrons
	xx[ii] = dx[ii] + xx[ii]
	# Changing velocity of the electrons
	u[ii] = u[ii] + 1
	# Checking which electrons have reached the anode
	anode = np.where(xx>n)[0]
	# If the electrons have reached anode setting all values to 0
	xx[anode] = 0
	u[anode] = 0
	dx[anode] = 0
	# Checking which velocities are above treshold
	treshvel = np.where(u>u0)[0]
	# Out of the excited ones only <=p of them are ionized
	ll = np.where(np.random.rand(len(treshvel))<=p)[0]
	# kl contains indices of the electrons which now suffer inelastic collisions
	kl = treshvel[ll]
	# Setting all the values to 0
	u[kl] = 0
	# Changing position of the collided electron to a random position 
	rho = random.uniform(0, 1)
	# Multiplying any random number rho generated uniformly between 0 and 1
	xx[kl] = xx[kl] - rho*dx[kl]
	# Excited atoms result in photon emission, so we have to add a photon at that point
	I.extend((xx[kl].T).tolist())
	
	m = int(np.random.randn()*Msig+M)
	# Checking where the slots are empty
	empty = np.where(xx==0)[0]
	# Filling the slots only such that the number of randomly selected slots
	# is lesser than the number of empty slots
	m = min(m, len(empty))
	# Inititializing the values of the newly introduced electrons as 1
	xx[empty[:m]] = 1
	# Initializing other values at these slots as 0
	u[empty[:m]] = 0
	dx[empty[:m]] = 0
	# Resetting the values of ii at the end of the loop to be used in next loop
	ii = np.where(xx>0)
	# Adding new values to final X and V vectors
	X.extend(xx[ii].tolist())
	V.extend(u[ii].tolist())

# Plotting histograms of I, V, X
# Plotting population density

plt.figure(0)
plt.title("Population density plot")
plt.xlabel(r'$n$ $\longrightarrow$')
plt.ylabel(r'$X$ $\longrightarrow$')
plt.grid(True)
plt.hist(X, bins = n)
plt.show()

# Plotting light intensity
plt.figure(1)
plt.title("Light Intensity Plot")
plt.grid(True)
plt.xlabel(r'$n$ $\longrightarrow$')
plt.ylabel(r'$Intensity$ $\longrightarrow$')
(pop, _bin, _un) = plt.hist(I, bins = n, density=True, edgecolor="black", rwidth=1)
plt.show()

# Electron phase space
plt.figure(2)
plt.title("Electron Phase Space")
plt.xlabel(r'$X$ $\longrightarrow$')
plt.ylabel(r'$V$ $\longrightarrow$')
plt.grid(True)
plt.plot(X, V, 'x')
plt.show()


# Printing out the intensity data
xpos = 0.5*(_bin[0:-1]+_bin[1:])
print("Intensity Data")
print("xpos\tcount")
for i in range(n):
	try:
		print("%f\t%f" %(xpos[i], pop[i]))
	except IndexError:
		print("End of code")








