from pylab import*
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import matplotlib.pyplot as plt
from sys import argv, exit
import math

Nx=25
Ny=25
radius=8
Niter=1000

# Block to ascertain the inputs given by the user are correct
# It also specifies the way the inputs given are interpreted
if len(argv)==1:
	print("Default values have been assigned")

elif len(argv)==2:
	print("Nx has been assigned %s" %argv[1])
	Nx=int(argv[1])
	Ny=Nx

elif len(argv)==3:
	if argv[1]!=argv[2]:
		print("Please enter equal values for Nx and Ny")
		exit(0)

	print("Nx has been assigned %s" %argv[1])
	print("Ny has been assigned %s" %argv[2])
	Nx=int(argv[1])
	Ny=int(argv[2])

elif len(argv)==4:
	if argv[1]!=argv[2]:
		print("Please enter equal values for Nx and Ny")
		exit(0)
	if int(argv[3])>int(argv[1]):
		print("Warning: Radius greater than the axes specified")
	print("Nx has been assigned %s" %argv[1])
	print("Ny has been assigned %s" %argv[2])
	print("radius has been assigned %s" %argv[3])
	Nx=int(argv[1])
	Ny=int(argv[2])
	radius=float(argv[3])

elif len(argv)==5:
	if argv[1]!=argv[2]:
		print("Please enter equal values for Nx and Ny")
		exit(0)
	if int(argv[3])>int(argv[1]):
		print("Warning: Radius greater than the axes specified")
	print("Nx has been assigned %s" %argv[1])
	print("Ny has been assigned %s" %argv[2])
	print("radius has been assigned %s" %argv[3])
	print("Niter has been assigned %s" %argv[4])
	Nx=int(argv[1])
	Ny=int(argv[2])
	radius=float(argv[3])
	Niter=int(argv[4])

else:
	print("Usage: %s <Nx> <Ny> <radius> <Niter>" %argv[0])
	exit(0)


# Declaring phi matrix
phi=np.zeros((Ny, Nx))
x = np.linspace(-floor(Nx/2), floor(Nx/2), Nx)
y = np.linspace(-floor(Ny/2), floor(Ny/2), Ny)
Y,X = np.meshgrid(y,x)

ii = np.where(Y**2+X**2<(radius**2))
phi[ii]=1
ii=ii-floor(Nx/2)

# Declaring errors array and a copy of phi
errors = np.zeros(Niter)
phinew = phi.copy()

# Plotting initial 3D plot of potential
f1 = plt.figure(1)
ax = p3.Axes3D(f1)
plt.title('Initial 3-D surface plot of the potential')
surf = ax.plot_surface(X/Nx, Y/Ny, phi.T, rstride=1, cstride=1, cmap=cm.jet)
plt.show()

# Initializing matrices for Calculating currents in the plate
Jx = np.zeros(phi.shape)
Jy = np.zeros(phi.shape)


# Finding phi after repeated iterations in this loop
for i in range(Niter):
	phinew[1:-1,1:-1] = 0.25*(phi[1:-1,0:-2]+phi[1:-1,2:]+phi[0:-2,1:-1]+phi[2:,1:-1])
	# Finding the error
	errors[i] = np.amax(abs(phinew-phi))
	ii =np.where(Y**2+X**2<(radius**2))
	phinew[ii]=1
	ii=ii-floor(Nx/2)
	phi = phinew.copy()
	# Applying boundary conditions
	# Left boundary
	phinew[1:-1,0] = phinew[1:-1,1]
	# Right boundary
	phinew[1:-1,-1] = phinew[1:-1,-2]
	# Top boundary
	phinew[0,1:-1] = phinew[1,1:-1]
	# Bottom Boundary
	phinew[-1,:] = 0

	for k in range(50):
		Jx[:,1:-1] = -0.5*(phi[:, 2:]-phi[:, :-2])
		# Left boundary and right boundary
		Jx[:,0] = 0
		Jx[:,-1] = 0
		Jy[1:-1,:] = -0.5*(phi[2:, :]-phi[:-2, :])
		# Top boundary and Bottom Boundary
		Jy[0,:] = 0
		Jy[-1,:] = 0
	
	# Plotting every 100th curve
	if i%100==0:
		plt.plot(X, Y, '.k', ms=1)
		plt.plot()
		plt.plot(ii[0],ii[1], 'ro')
		plt.title(r'Plot of $\phi$ contours at $%dth$ iteration' %i)
		plt.contour(X, Y, phinew.T)
		plt.xlabel(r"$x$ $\longrightarrow$")
		plt.ylabel(r"$y$ $\longrightarrow$")
		locs, labels = plt.xticks()
		labels = [round(float(item)*(1/Nx),2) for item in locs]
		plt.xticks(locs[1:-1], labels[1:-1], rotation=90)
		locs, labels = plt.yticks()
		labels = [round(float(item)*(1/Ny),2) for item in locs]
		plt.yticks(locs[1:-1], labels[1:-1])
		plt.tight_layout()
		plt.show()

		plt.quiver(-y, -x, -Jx, -Jy)
		plt.plot(ii[0],ii[1], 'ro')
		plt.xlabel(r"$x$ $\longrightarrow$")
		plt.ylabel(r"$y$ $\longrightarrow$")
		locs, labels = plt.xticks()
		labels = [round(float(item)*(1/Nx),2) for item in locs]
		plt.xticks(locs[1:-1], labels[1:-1], rotation=90)
		locs, labels = plt.yticks()
		labels = [round(float(item)*(1/Ny),2) for item in locs]
		plt.yticks(locs[1:-1], labels[1:-1])
		plt.title(r'Plot of $J$ at $%dth$ iteration' %i)
		plt.show()
	

iter_array = np.linspace(1, Niter, Niter)
# Finding the fits using least squares approximation
ln_err1 = np.log(errors)
col1 = np.ones(Niter)
# Stacking the columns for approximating values through linear regression
A1 = np.column_stack((col1, iter_array))
ls_co1 = np.linalg.lstsq(A1, ln_err1, rcond=None)[0]
A1 = np.exp(ls_co1[0])
B1 = ls_co1[1]
error_est1 = A1*np.exp(B1*iter_array)
# Printing first error estimate
print("Approximation with all the data points taken into account")
print("A1: %f" %A1)
print("B1: %f" %B1)

ln_err2 = np.log(errors)[500:]
# Using column stack to append columns side by side for easy evaluation of the coefficients
A2 = np.column_stack((col1[500:], iter_array[500:]))
# Linear regression to find coefficients
ls_co2 = np.linalg.lstsq(A2, ln_err2, rcond=None)[0]
A2 = np.exp(ls_co2[0])
B2 = ls_co2[1]
error_est2 = A2*np.exp(B2*iter_array)
# Printing second error estimate using points after 500yh satmp
print("Approximation with all the data points after 500 taken into account")
print("A2: %f" %A2)
print("B2: %f" %B2)


# Graphing the errors
plt.title("Error plot")
plt.yscale('log')
plt.xscale('log')
plt.grid(linestyle=':')
# Plotting actual errors
plt.plot(iter_array, errors, label = 'Actual Errors')
plt.legend()
# Plotting error estimate 1
plt.plot(iter_array[::50], error_est1[::50], 'ro', label='Error Estimate 1')
plt.legend()
# Plotting error estimate 2
plt.plot(iter_array[::50], error_est2[::50], 'go', label='Error Estimate 2')
plt.legend()
plt.xlabel(r"Number of iterations $\longrightarrow$")
plt.ylabel(r"Errors $\longrightarrow$")
plt.show()


# Graph of the final potential in 3D
f1 = plt.figure(1)
ax = p3.Axes3D(f1)
plt.title('The 3-D surface plot of the potential after %d iterations' %Niter)
surf = ax.plot_surface(Y/Ny, X/Nx, phi.T, rstride=1, cstride=1, cmap=cm.jet)
plt.show()























