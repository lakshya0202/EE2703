from sys import argv, exit, maxsize
import numpy as np
import cmath
import math

'''
Important Note:
The ground must be denoted as 0 or GND.
The code works for only one AC voltage source
'''

'''
Declaring classes for all possible emelents of the circuit:
VCVS: Voltage Controlled Voltage Source
VCCS: Voltage controlled Current Source
CCVS: Current Controlled Voltage Source
CCCS: Current Controlled Current Source
'''
class Resistor:
	def __init__(self, node1, node2, value):
		self.node1=int(node1)
		self.node2=int(node2)
		self.value=float(value)

	def conductance(self, f=0):
		return 1/self.value

class Inductor:
	def __init__(self, node1, node2, value):
		self.node1=int(node1)
		self.node2=int(node2)
		self.value=float(value)

	def conductance(self, f=0):
		if f==0:
			return complex(maxsize)
		return complex(0, -1/(2*math.pi*f*self.value))

class Capacitor:
	def __init__(self, node1, node2, value):
		self.node1=int(node1)
		self.node2=int(node2)
		self.value=float(value)

	def conductance(self, f=0):
		return complex(0, 2*math.pi*f*self.value)

class Vs:
	def __init__(self, node1, node2, value):
		self.node1=int(node1)
		self.node2=int(node2)
		self.value=float(value)

	def voltage_value(self):
		return self.value

class Vs_sine:
	def __init__(self, node1, node2, value, ac_dc="", phase=0):
		self.node1=int(node1)
		self.node2=int(node2)
		self.value=float(value)/2
		self.ac_dc=ac_dc
		self.phase=float(phase)

	def voltage_value(self):
		if self.ac_dc=='ac':
			return cmath.rect(self.value, self.phase)
		else:
			return self.value

class Is:
	def __init__(self, node1, node2, value):
		self.node1=int(node1)
		self.node2=int(node2)
		self.value=float(value)

class VCVS:
	def __init__(self, node1, node2, node3, node4, value):
		self.node1=int(node1)
		self.node2=int(node2)
		self.node3=int(node3)
		self.node4=int(node4)
		self.value=float(value)

class VCCS:
	def __init__(self, node1, node2, node3, node4, value):
		self.node1=int(node1)
		self.node2=int(node2)
		self.node3=int(node3)
		self.node4=int(node4)
		self.value=float(value)

class CCVS:
	def __init__(self, node1, node2, V, value):
		self.node1=int(node1)
		self.node2=int(node2)
		self.V=float(V)
		self.value=float(value)

class CCCS:
	def __init__(self, node1, node2, V, value):
		self.node1=int(node1)
		self.node2=int(node2)
		self.V=float(V)
		self.value=float(value)


# Function to assign integer node values
def node_value(k):
	global nodes
	global key_count
	if k in nodes:
		return nodes[k]
	else:
		nodes[k] = key_count
		key_count+=1
		return nodes[k]

CIRCUIT=".circuit"
END=".end"
AC=".ac"
# components stores all the distict elements of the circuit as objects of that class
components={}
# Stores the maximum node value, i.e., number of nodes
max_node=0
# No of voltage sources
m=0

# Makes sure user inputs the appropriate netlist file while
# running the code
if len(argv)!=2:
	print("Usage: %s <inputfile>" %argv[0])
	exit(0)

# Holds count of all nodes encountered by the below loop
key_count=1
# Dictionary to hold corresponding values of nodes
nodes={'GND':0, '0':0, 'gnd':0}
ground_count=0
# Variable for holding a frequency source in case one is present
freq=0
# Voltage sources count
# try catch block for looping through all 
# components sepcified by the netlist
try:
	with open(argv[1]) as f:
		lines=f.readlines()
		start=-1; end=-2;
		ac=0
		for line in lines:
			if CIRCUIT == line[:len(CIRCUIT)]:
				start=lines.index(line)
			elif END== line[:len(END)]:
				end=lines.index(line)
				break
		if start>=end:
			print("Invalid circuit definition")
			exit(0)
		
		if len(lines)>=end+2:
			for line in lines[end+1:]:
				if line.split()[0]==AC:
					l=line.split('#')[0]
					l=l.split()
					if len(l)!=3:
						print("Invalid circuit definition")
						exit(0)
					if lines.index(line)==end+1:
						freq=l[2]
					if freq!=l[2]:
						print("Multiple frequencies cannot be handled")
						exit(0)

				else:
					print("Invalid circuit definition")
					exit(0)

		for line in lines[start+1:end]:
			l=line.split('#')[0]
			l=l.split()
			# For all these cases Node values are assumed to be integers
			# except the presence of GND, which is substituted with 0
			if line[0]=='R':
				if len(l)!=4:
					print("Invalid definition of Resistor")
					exit(0)
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				components[l[0]]=Resistor(l[1], l[2], l[3])
				max_node=max(components[l[0]].node1, components[l[0]].node2, max_node)

			elif line[0]=='L':
				if len(l)!=4:
					print("Invalid definition of Inductor")
					exit(0)
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				components[l[0]]=Inductor(l[1], l[2], l[3])
				max_node=max(components[l[0]].node1, components[l[0]].node2, max_node)

			elif line[0]=='C':
				if len(l)!=4:
					print("Invalid definition of Capacitor")
					exit(0)
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				components[l[0]]=Capacitor(l[1], l[2], l[3])
				max_node=max(components[l[0]].node1, components[l[0]].node2, max_node)
			
			elif line[0]=='V' and freq==0:
				if len(l)!=4:
					print("Invalid definition of dc voltage source")
					exit(0)
				m=m+1
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				components[l[0]]=Vs(l[1], l[2], l[3])
				max_node=max(components[l[0]].node1, components[l[0]].node2, max_node)
			
			elif freq!=0 and line[0]=='V':
				if len(l)!=6 and len(l)!=5:
					print("Invalid circuit definition")
					exit(0)
				
				m=m+1
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				if l[3]=='ac' :
					components[l[0]]=Vs_sine(l[1], l[2], l[4], l[3], float(l[5]))
				elif l[3]=='dc':
					print("Invalid circuit - Cannot handle both DC and AC")
					exit(0)
					components[l[0]]=Vs_sine(l[1], l[2], l[4], l[3], 0)
				else:
					print("Incorrect definition of Voltage source")
					exit(0)
				max_node=max(components[l[0]].node1, components[l[0]].node2, max_node)

			elif line[0]=='I':
				if len(l)!=4:
					print("Invalid definition of Current Source")
					exit(0)
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				components[l[0]]=Is(l[1], l[2], l[3])
				max_node=max(components[l[0]].node1, components[l[0]].node2, max_node)

			elif line[0]=='E':
				if len(l)!=6:
					print("Invalid definition of Voltage Controlled Voltage Source")
					exit(0)
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				l[3] = node_value(l[3])
				l[4] = node_value(l[4])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				if l[3]==0 :
					ground_count+=1
				if l[4]==0:
					ground_count+=1
				components[l[0]]=VCVS(l[1], l[2], l[3], l[4], l[5])
				max_node=max(components[l[0]].node1, components[l[0]].node2, components[l[0]].node3, components[l[0]].node4, max_node)

			elif line[0]=='G':
				if len(l)!=6:
					print("Invalid definition of Voltage Controlled Current Source")
					exit(0)
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				l[3] = node_value(l[3])
				l[4] = node_value(l[4])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				if l[3]==0 :
					ground_count+=1
				if l[4]==0:
					ground_count+=1
				components[l[0]]=VCCS(l[1], l[2], l[3], l[4], l[5])
				max_node=max(components[l[0]].node1, components[l[0]].node2, components[l[0]].node3, components[l[0]].node4, max_node)

			elif line[0]=='H':
				if len(l)!=5:
					print("Invalid definition of Current Controlled Voltage Source")
					exit(0)
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				components[l[0]]=CCVS(l[1], l[2], l[3], l[4])
				max_node=max(components[l[0]].node1, components[l[0]].node2, max_node)

			elif line[0]=='F':
				if len(l)!=4:
					print("Invalid definition of Current Controlled Current Source")
					exit(0)
				
				l[1] = node_value(l[1])
				l[2] = node_value(l[2])
				if l[1]==0 :
					ground_count+=1
				if l[2]==0:
					ground_count+=1
				components[l[0]]=CCCS(l[1], l[2], l[3], l[4])
				max_node=max(components[l[0]].node1, components[l[0]].node2, max_node)	

			else:
				print('Component not defined / Not a valid component')
				exit(0)

except IOError:
	print("Invalid file")
	exit(0)

# Ground should be represented by the following to show distinguishability
if ground_count==0:
	print("Please denote ground by 0 or GND or gnd")
	exit(0)

# key value list of nodes dictionary 
# used to append with the voltage symbol to show the voltage at that point
nodes_key_list = list(nodes.keys())
nodes_val_list = list(nodes.values())

# n denotes the number of nodes
# m denotes the number of voltage sources
n=max_node

# Declaring all the matrices required for finding node voltages and currents
G=np.zeros((n,n)).astype('complex')
B=np.zeros((n,m)).astype('complex')
C=np.zeros((m,n)).astype('complex')
D=np.zeros((m,m)).astype('complex')
i=np.zeros((n,1)).astype('complex')
e=np.zeros((m,1)).astype('complex')
# Holds the name of the corresponding voltage source
# for refernce later
j=[]
# Holds the representation of the nodal voltage at that point
j_fin=[]
# Array storing symbols of all voltages for n nodes
v=[]
# Loop to declare node voltage names
for k in range(1,n+1):
	pos = nodes_val_list.index(k)
	v.append('V_'+str(nodes_key_list[pos]))

# Needed to keep count of all voltage sources parsed
vsCount=0;

# Loop for writing the coductance matrix equations
# This part is inspired by the SCAM MODEL
for key in components:
	fc=key[0]
	n1=components[key].node1
	n2=components[key].node2

	if fc=='R' or fc=='L' or fc=='C':
		if n1==0:
			G[n2-1, n2-1]=G[n2-1, n2-1]+components[key].conductance(freq)
		elif n2==0:
			G[n1-1, n1-1]=G[n1-1, n1-1]+components[key].conductance(freq)
		else:
			G[n1-1, n1-1]=G[n1-1, n1-1]+components[key].conductance(freq)
			G[n2-1, n2-1]=G[n2-1, n2-1]+components[key].conductance(freq)
			G[n1-1, n2-1]=G[n1-1, n2-1]-components[key].conductance(freq)
			G[n2-1, n1-1]=G[n2-1, n1-1]-components[key].conductance(freq)

	elif fc=='V':
		# Here n2 is the cathode and n1 is anode
		vsCount=vsCount+1
		e[vsCount-1]=components[key].voltage_value()
		j.append(key)
		j_fin.append('I_'+key)
		if n1!=0:
			B[n1-1, vsCount-1]=B[n1-1, vsCount-1]+1
			C[vsCount-1, n1-1]=C[vsCount-1,n1-1]+1
		if n2 !=0:
			B[n2-1, vsCount-1]=B[n2-1, vsCount-1]-1
			C[vsCount-1, n2-1]=C[vsCount-1, n2-1]-1

	elif fc=='I':
		# Here current flows from n1 to n2
		if n1!=0:
			i[n1-1]=i[n1-1]-components[key].value
		if n2!=0:
			i[n2-1]=i[n2-1]+components[key].value

	elif fc=='E':
		# This is the VCVS case
		vsCount=vsCount+1
		n3=components[key].node3
		n4=components[key].node4
		j.append(key)
		j_fin.append('I_'+key)
		if n1!=0:
			B[n1-1, vsCount-1]=B[n1-1, vsCount-1]+1
			C[vsCount-1, n1-1]=C[vsCount-1, n1-1]+1
		if n2!=0:
			B[n2-1, vsCount-1]=B[n2-1, vsCount-1]+1
			C[vsCount-1, n2-1]=C[vsCount-1, n2-1]+1
		if n3!=0:
			C[vsCount-1, n3-1]=C[vsCount-1, n3-1]-components[key].value
		if n4!=0:
			C[vsCount-1, n4-1]=C[vsCount-1, n4-1]+components[key].value

	elif fc=='G':
		# This is the VCCS case
		# Positive side of Control voltage
		n3=components[key].node3
		# negative side of control voltage
		n4=components[key].node4
		mystr=""
		for k in [n1, n2, n3 ,n4]:
			if k!=0:
				mystr+='1'
		if mystr=="0000" or mystr=="0011" or mystr=="0001" or mystr=="0010" or mystr=="0100" or mystr=="1000" or mystr=="1100":
			print("ERROR IN VCCS")
			exit(0)
		elif mystr=='1111':
			G[n1-1, n3-1]=G[n1-1, n3-1]+components[key].value
			G[n1-1, n4-1]=G[n1-1, n4-1]-components[key].value
			G[n2-1, n3-1]=G[n2-1, n3-1]-components[key].value
			G[n2-1, n4-1]=G[n2-1, n4-1]+components[key].value
		elif mystr=='0111':
			G[n2-1, n3-1]=G[n2-1, n3-1]-components[key].value
			G[n2-1, n4-1]=G[n2-1, n4-1]+components[key].value
		elif mystr=='0101':
			G[n2-1, n4-1]=G[n2-1, n4-1]+components[key].value
		elif mystr=='0110':
			G[n2-1, n3-1]=G[n2-1, n3-1]-components[key].value
		elif mystr=='1011':
			G[n1-1, n3-1]=G[n1-1, n3-1]+components[key].value
			G[n1-1, n4-1]=G[n1-1, n4-1]-components[key].value
		elif mystr=='1001':
			G[n1-1, n4-1]=G[n1-1, n4-1]-components[key].value
		elif mystr=='1010':
			G[n1-1, n3-1]=G[n1-1, n3-1]+components[key].value
		elif mystr=='1101':
			G[n1-1, n4-1]=G[n1-1, n4-1]-components[key].value
			G[n2-1, n4-1]=G[n2-1, n4-1]+components[key].value
		elif mystr=='1110':
			G[n1-1, n3-1]=G[n1-1, n3-1]+components[key].value
			G[n2-1, n3-1]=G[n2-1, n3-1]-components[key].value

	#elif fc=='F':
		# Since for CCCS we need the controlling current
		# which is defines as the current through one of the voltage sources
		# Since this voltage source may not be defined yet, i.e., it comes 
		# later in the circuit definition, we add this in the matrix later

	elif fc=='H':
		# CCVS case
		# Since this also requires the controlling current which may not
		# have been defined yet, we leave this for later
		vsCount=vsCount+1
		j.append(key)
		j_fin.append('I_'+key)
		if n1!=0:
			B[n1-1, vsCount-1]=B[n1-1, vsCount-1]+1
			C[vsCount-1, n1-1]=C[vsCount-1, n1-1]+1
		if n2!=0:
			B[n2-1, vsCount-1]=B[n2-1, vsCount-1]-1
			C[vsCount-1, n2-1]=C[vsCount-1, n1-1]-1

'''
Now that all the voltage sources have been parsed, we can
go through and finish off the CCVS and CCCS elements which depend on the
current through those sources.
'''
for key in components:
	fc=key[0]
	n1=components[key].node1
	n2=components[key].node2

	if fc=='H':
		hInd = j.index(key)
		cv = components[key].V
		cvInd = j.index(cv)
		D[hInd, cvInd]=D[hInd, cvInd]-components[key].value

	elif fc=='F':
		cv=components[key].V
		cvInd=j.index(cv)
		if n1!=0:
			B[n1, cvInd]=B[n1, cvInd]+components[key].value
		if n2!=0:
			B[n2, cvInd]=B[n2, cvInd]-components[key].value

'''
Submatrices are complete
Now need to form A(complete conductance matrix), x and z
'''
shG = np.shape(G)
shB = np.shape(B)
shC = np.shape(C)
shD = np.shape(D)

# First horizontal combination of G and B
r1 = shG[0]
c1 = shG[1]+shB[1]

GB=np.zeros((r1, c1)).astype('complex')
GB[0:r1, 0:shG[1]]=G
GB[0:r1, shG[1]:c1]=B

# Second horizontal combination of C and D
r2 = shC[0]
c2 = shC[1]+shD[1]

CD=np.zeros((r2, c1)).astype('complex')
CD[0:r2, 0:shC[1]]=C
CD[0:r2, shC[1]:c2]=D

# Now vertical combination
row_tot=n+m
A = np.zeros((r1+r2, c1)).astype('complex')
A[0:n, 0:c1]=GB
A[n:row_tot, 0:c1]=CD
 
# Horizontal combination of v and j_fin
# to form x column vector of unknown variables
x=[]
for k in range(n):
	x.append(v[k])
for k in range(m):
	x.append(j_fin[k])

z=np.zeros((n+m,1)).astype('complex')
z[0:n,0]=i[0:n,0]
z[n:n+m,0]=e[0:m,0]

x_fin=np.linalg.solve(A, z)
for i in range(n+m):
	if x_fin[i][0].imag==0:
		ans = x_fin[i][0].real
	else:
		ans = x_fin[i][0]
	# Prints out the final values with corresponding notations
	print(str(x[i])+" = "+str(ans))
