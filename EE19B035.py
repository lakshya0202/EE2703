import matplotlib.pyplot as plt
import numpy as np
import warnings


# Radius of the loop
# It is considered to be in centimeters
a = 10
'''
Q3
'''
# Breaking the loop in 100 sections
# I have assumed phi spans across 360 degreed alsong the plane
# Sir has used both polar and cylindrical coordinates representations
# Declaring the appropriate constants
# Added to phi to centre the point of consideration
d_theta = (np.pi)/100
# Constant mu0
mu0 = 4*np.pi*(10**(-7))
# Creating the phi array conating points at centres of segments
# all around a circle
phi = np.linspace(d_theta,2*np.pi-d_theta, 100)
# Declaring the array I
I = ((4*np.pi)/mu0)*np.cos(phi)
# Getting the appropriate points for the x coordinates
x = a*np.cos(phi)
# Getting the appropriate points for the y coordinates
y = a*np.sin(phi)
# Plotting the figure
fig = plt.figure()
# Creating a 3d projection subplot
ax = fig.add_subplot(111, projection='3d')
ax.title.set_text('Plot of current elements at t=0')
ax.plot(x, y, I, 'bo', markersize = 5)
ax.plot(x, y, np.amin(I), 'r', markersize = 1)
ax.legend(['Variation of current magnitude', 'Wire loop'])
ax.grid(True)
plt.show()

# Creating a quiver plot for the currents
# Creating the meshgrid with x and y values
YY, XX = np.meshgrid(x, y, sparse =False)
# Assigning the value for the current matrix
I = ((4*np.pi)/mu0)*(YY/a)
# Setting all points other than those on the radius as 0
I[np.where(np.sqrt(XX**2+YY**2)!=a)] = 0
# Declaring the x direction of the current
Ix = -I*(XX)/a
# Declaring the y direction of current
Iy = I*(YY)/a
# Plotting the quiver plot of the currents
plt.figure()
plt.title("Surface plot of the currents")
plt.quiver(x, y, Ix, Iy)
plt.show()

'''
Q4
'''
# Declaring possible values of xi and yj to find the magnetic field around it 
delta_x = 1
delta_y = 1
# MESHGRID
x_mesh = np.linspace(-delta_x, delta_x, 3)
y_mesh = np.linspace(-delta_y, delta_y, 3)
z_mesh = np.linspace(1, 1000, 1000)
Y,X,Z = np.meshgrid(x_mesh, y_mesh, z_mesh)
# Obtaining r' vector
r_ = np.column_stack((x,y))
# Obtaining the dl' vector
dl = (a*2*np.pi)/100
dlx = dl*(-np.sin(phi))
dly = dl*(np.cos(phi))
#dl_ = np.column_stack((dlx, dly))

dl_ = np.zeros(r_.shape)
dl_[:-1, :] = r_[1:,:]-r_[:-1,:]
dl_[-1,:] = r_[0,:] - r_[-1, :]

# Declaring B array
B = np.zeros(1000)


'''
Q5, Q6
'''
# Function to return the vector A(ijk)
def calc(l):
	# Finding Rijkl matrix for each segment l
	Rijkl = np.sqrt(np.multiply(X-r_[l,0], X-r_[l,0]) + np.multiply(Y-r_[l,1], Y-r_[l,1]) + np.multiply(Z,Z))
	# Declaring a complex variable jk
	jk = complex(0, 0.1)
	# Multiplying the complex term with Rijkl
	exp_term = jk*Rijkl
	# np.divide achieves element-wise division
	# Finding the x component of the potential
	Axijk = np.divide((np.cos(phi[l]))*dl_[l,0]*(np.exp(-exp_term)), Rijkl)
	# Finding the y component of the potential
	Ayijk = np.divide((np.cos(phi[l]))*dl_[l,1]*(np.exp(-exp_term)), Rijkl)
	# Returning the x and y components of the potential
	return Axijk, Ayijk

'''
I am using a loop of 100 iterations here because the value of 
the magnetic field at each of these points should be calculated separately
depending on the part of the wire which contributes to it
'''
'''
Q8
'''
# Declaring the arrays to store the x and y components of the Aijk
Ax_f = np.zeros(Y.shape)
Ay_f = np.zeros(Y.shape)
# for loop for each small length of the ring
for i in range(100):
	# Assigning Ax and Ay the values retured by calc
	Ax_, Ay_  = calc(i)
	# Adding the returned values to the final matrices
	Ax_f = np.add(Ax_f, Ax_)
	Ay_f = np.add(Ay_f, Ay_)


# Another way to fing the magnetic field
# Using the where command
'''
x1 = np.where((Y==0) & (X==1))
y1 = np.where((X==0) & (Y==1))
x1_ = np.where((Y==0) & (X==-1))
y1_ = np.where((X==0) & (Y==-1))

dAy = Ay_f[x1]
dAx = Ax_f[y1]
dAy_ = Ay_f[x1_]
dAx_ = Ax_f[y1_]

B = (dAy - dAx - dAy_ + dAx_)/4.0
'''
# Using matrix manipulation to find the final B value
# Ay_f[2,1,:] : x=1, y=0
# Ax_f[1,2,:] : x=0, y=1
# Ay_f[0,1,:] : x = -1, y=0
# Ax_f[1,0,:] : x=0, y=-1
# Only these z axis values are chosen for the computation
B = (Ay_f[2,1,:]-Ax_f[1,2,:]-Ay_f[0,1,:]+Ax_f[1,0,:])/4
# The compiler displays a worning when it ignores the imaginary oart while plotting 
# the magnetic field. This command filters those warnings
warnings.filterwarnings("ignore")
'''
Q9
Plotting the Magnetic field obtained
'''
# Declaring the figure where the plot is to appear
plt.figure(1)
# Declaring the title of the plot
plt.title("Loglog Plot of B")
# Declaring the type of plot, in this case it is loglog
plt.loglog(np.linspace(1,1000,1000), np.abs(B), 'o')
# Labelling the x axis
plt.xlabel(r"log(z) $\longrightarrow$")
# Labelling the y axis
plt.ylabel(r"log(|B|) $\longrightarrow$")
# Option for making grid appear for readability
plt.grid(True)
# Finally plot is displayed until exited
plt.show()

# Declaring the figure where the plot is to appear
plt.figure(2)
# Declaring the title of the plot
plt.title("Plot of B")
# Declaring the type of plot
plt.plot(np.linspace(1,1000,1000), B, 'o')
# Labelling the x axis
plt.xlabel(r"z $\longrightarrow$")
# Labelling the y axis
plt.ylabel(r"B $\longrightarrow$")
# Option for making grid appear for readability
plt.grid(True)
# Finally plot is displayed until exited
plt.show()

'''
Q10
Fitting the curve using lstsq
'''
# Declaring z array
z = np.linspace(1,1000,1000)
# Fing the log of z for least squares manipulation
ln_z = np.log(z)
# Finding the log of B for least squares manipulation
ln_B = np.log(np.abs(B))
# Declaring a column of ones which represents the constant log c 
# multiplied to it
col1 = np.ones(1000)
# Stacking the columns side by side to create the least squares matrix
A = np.column_stack((col1, ln_z))
# Applying the lstsq method from linalg and taking the first argument
lsq = np.linalg.lstsq(A, ln_B, rcond=None)[0]
# Taking the exponent value for finding c
c = np.exp(lsq[0])
# b is used as is as it was a part of z^b
b = lsq[1]
# Finding the estimate using the constants found
B_est = c*(np.power(z, b))

# Plotting the estimate graph
plt.figure(1)
# Title of the graph
plt.title("Loglog Plot of estimate of B")
# Using the loglog option
plt.loglog(z, np.abs(B), 'o')
plt.loglog(z, np.abs(B_est), 'ro')
# Adding the legend for the graph
plt.legend(['Actual Value','Estimate'])
# Labelling the x axis
plt.xlabel(r"log(z) $\longrightarrow$")
# Labelling the y axis
plt.ylabel(r"log(|B|) $\longrightarrow$")
# Option for making grid appear for readability
plt.grid(True)
# Finally plot is displayed until exited
plt.show()

# Plotting the estimate graph
plt.figure(2)
# Title of the graph
plt.title("Plot of estimate of B")
# Using the normal plot option
plt.plot(z, np.abs(B), 'o')
# Plotting both the estimate and actual values
plt.plot(z, np.abs(B_est), 'ro')
# Adding the legend for the graph
plt.legend(['Actual Value','Estimate'])
# Labelling the x axis
plt.xlabel(r"z $\longrightarrow$")
# Labelling the y axis
plt.ylabel(r"B $\longrightarrow$")
# Option for making grid appear for readability
plt.grid(True)
# Finally plot is displayed until exited
plt.show()

'''
Q11
'''
# Here the the exponent value b is printed to the output window
print("The rate at which B falls off is %f" %b)





'''
Doing the same evaluation but taking absolute values of the current components
'''
'''
Q5, Q6
'''
# Function to return the vector A(ijk)
def calc(l):
	# Finding Rijkl matrix for each segment l
	Rijkl = np.sqrt(np.multiply(X-r_[l,0], X-r_[l,0]) + np.multiply(Y-r_[l,1], Y-r_[l,1]) + np.multiply(Z,Z))
	# Declaring a comlex variable jk
	jk = complex(0, 0.1)
	# Multiplying the complex term with Rijkl
	exp_term = jk*Rijkl
	# np.divide achieves element-wise division
	# Finding the x component of the potential
	Axijk = np.divide(abs(np.cos(phi[l]))*dl_[l,0]*(np.exp(-exp_term)), Rijkl)
	# Finding the y component of the potential
	Ayijk = np.divide(abs(np.cos(phi[l]))*dl_[l,1]*(np.exp(-exp_term)), Rijkl)
	# Returning the x and y components of the potential
	return Axijk, Ayijk

'''
I am using a loop of 100 iterations here because the value of 
the magnetic field at each of these points should be calculated separately
depending on the part of the wire which contributes to it
'''
'''
Q8
'''
# Declaring B array
B = np.zeros(1000)
# Declaring the arrays to store the x and y components of the Aijk
Ax_f = np.zeros(Y.shape)
Ay_f = np.zeros(Y.shape)
# for loop for each small length of the ring
for i in range(100):
	# Assigning Ax and Ay the values retured by calc
	Ax_, Ay_  = calc(i)
	# Adding the returned values to the final matrices
	Ax_f = np.add(Ax_f, Ax_)
	Ay_f = np.add(Ay_f, Ay_)


# Using matrix manipulation to find the final B value
B = (Ay_f[2,1,:]-Ax_f[1,2,:]-Ay_f[0,1,:]+Ax_f[1,0,:])/4
warnings.filterwarnings("ignore")
'''
Q9
Plotting the Magnetic field obtained
'''
# Declaring the figure where the plot is to appear
plt.figure(1)
# Declaring the title of the plot
plt.title("Loglog Plot of B")
# Declaring the type of plot, in this case it is loglog
plt.loglog(np.linspace(1,1000,1000), np.abs(B), 'o')
# Labelling the x axis
plt.xlabel(r"log(z) $\longrightarrow$")
# Labelling the y axis
plt.ylabel(r"log(|B|) $\longrightarrow$")
# Option for making grid appear for readability
plt.grid(True)
# Finally plot is displayed until exited
plt.show()

# Declaring the figure where the plot is to appear
plt.figure(2)
# Declaring the title of the plot
plt.title("Plot of B")
# Declaring the type of plot
plt.plot(np.linspace(1,1000,1000), B, 'o')
# Labelling the x axis
plt.xlabel(r"z $\longrightarrow$")
# Labelling the y axis
plt.ylabel(r"B $\longrightarrow$")
# Option for making grid appear for readability
plt.grid(True)
# Finally plot is displayed until exited
plt.show()

'''
Q10
Fitting the curve using lstsq
'''
# Declaring z array
z = np.linspace(1,1000,1000)
# Fing the log of z for least squares manipulation
ln_z = np.log(z)
# Finding the log of B for least squares manipulation
ln_B = np.log(np.abs(B))
# Declaring a column of ones which represents the constant log c 
# multiplied to it
col1 = np.ones(1000)
# Stacking the columns side by side to create the least squares matrix
A = np.column_stack((col1, ln_z))
# Applying the lstsq method from linalg and taking the first argument
lsq = np.linalg.lstsq(A, ln_B, rcond=None)[0]
# Taking the exponent value for finding c
c = np.exp(lsq[0])
# b is used as is as it was a part of z^b
b = lsq[1]
# Finding the estimate using the constants found
B_est = c*(np.power(z, b))

# Plotting the estimate graph
plt.figure(1)
# Title of the graph
plt.title("Loglog Plot of estimate of B")
# Using the loglog option
plt.loglog(z, np.abs(B), 'o')
plt.loglog(z, np.abs(B_est), 'ro')
# Adding the legend for the graph
plt.legend(['Actual Value','Estimate'])
# Labelling the x axis
plt.xlabel(r"log(z) $\longrightarrow$")
# Labelling the y axis
plt.ylabel(r"log(|B|) $\longrightarrow$")
# Option for making grid appear for readability
plt.grid(True)
# Finally plot is displayed until exited
plt.show()

# Plotting the estimate graph
plt.figure(2)
# Title of the graph
plt.title("Plot of estimate of B")
# Using the normal plot option
plt.plot(z, np.abs(B), 'o')
# Plotting both the estimate and actual values
plt.plot(z, np.abs(B_est), 'ro')
# Adding the legend for the graph
plt.legend(['Actual Value','Estimate'])
# Labelling the x axis
plt.xlabel(r"z $\longrightarrow$")
# Labelling the y axis
plt.ylabel(r"B $\longrightarrow$")
# Option for making grid appear for readability
plt.grid(True)
# Finally plot is displayed until exited
plt.show()

'''
Q11
'''
# Here the the exponent value b is printed to the output window
print("The rate at which B falls off is %f" %b)






