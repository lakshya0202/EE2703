from pylab import *
from scipy.linalg import lstsq
import numpy as np

'''
Q1
'''

# Spectrum of sin(sqrt(2)t)
t=linspace(-pi,pi,65)
t=t[:-1]
dt=t[1]-t[0]
fmax=1/dt
y=sin(sqrt(2)*t)
y[0]=0 
y=fftshift(y) 
Y=fftshift(fft(y))/64.0 
w=linspace(-pi*fmax,pi*fmax,65)
w=w[:-1]
figure(1)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(\sqrt{2}t)$") 
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-10,10])
ylabel(r"$\angle$$Y$",size=16) 
xlabel(r"$\omega$",size=16)
grid(True)
show()


# Plot of the time function over several time periods
t1=linspace(-pi,pi,65)
t1=t1[:-1] 
t2=linspace(-3*pi,-pi,65)
t2=t2[:-1] 
t3=linspace(pi,3*pi,65)
t3=t3[:-1]
figure(2) 
plot(t1,sin(sqrt(2)*t1),'b',lw=2) 
plot(t2,sin(sqrt(2)*t2),'r',lw=2) 
plot(t3,sin(sqrt(2)*t3),'r',lw=2) 
ylabel(r"$y$",size=16) 
xlabel(r"$t$",size=16) 
title(r"$\sin\left(\sqrt{2}t\right)$") 
grid(True)
show()

# Replicating the curve in blue
t1=linspace(-pi,pi,65);t1=t1[:-1] 
t2=linspace(-3*pi,-pi,65);t2=t2[:-1] 
t3=linspace(pi,3*pi,65);t3=t3[:-1] 
y=sin(sqrt(2)*t1)
figure(3)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'ro',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin(\sqrt{2}t)$ with $t$ wrapping every $2\pi$ ") 
grid(True)
show()

# Veryfying DFT for a digital ramp
t=linspace(-pi,pi,65)
t=t[:-1]
dt=t[1]-t[0]
fmax=1/dt
y=t
y[0]=0
y=fftshift(y)
Y=fftshift(fft(y))/64.0 
w=linspace(-pi*fmax,pi*fmax,65)
w=w[:-1]
figure(4) 
semilogx(abs(w),20*log10(abs(Y)),lw=2) 
xlim([1,10])
ylim([-20,0]) 
xticks([1,2,5,10],["1","2","5","10"],size=16) 
ylabel(r"$|Y|$ (dB)",size=16) 
title(r"Spectrum of a digital ramp") 
xlabel(r"$\omega$",size=16)
grid(True)
show()

# Now trying to suppress the jumps at the beginning and end of each window
# Hamming window is used to accomplish this
t1=linspace(-pi,pi,65)
t1=t1[:-1]
t2=linspace(-3*pi,-pi,65)
t2=t2[:-1] 
t3=linspace(pi,3*pi,65)
t3=t3[:-1] 
n=arange(64) 
wnd=fftshift(0.54+0.46*cos(2*pi*n/63)) 
y=sin(sqrt(2)*t1)*wnd
figure(5)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'ro',lw=2)
ylabel(r"$y$",size=16)
xlabel(r"$t$",size=16)
title(r"$\sin(\sqrt{2}t)\times w(t)$ with $t$ wrapping every $2\pi$ ") 
grid(True)
show()

# Finding the DFT of the above sequence
t=linspace(-pi,pi,65)
t=t[:-1] 
dt=t[1]-t[0]
fmax=1/dt
n=arange(64) 
wnd=fftshift(0.54+0.46*cos(2*pi*n/63)) 
y=sin(sqrt(2)*t)*wnd
y[0]=0 
y=fftshift(y) 
Y=fftshift(fft(y))/64.0 
w=linspace(-pi*fmax,pi*fmax,65)
w=w[:-1]
figure(6)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-8,8])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\sin(\sqrt{2}t)\times w(t)$") 
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-8,8])
ylabel(r"$\angle$$Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

# Trying the same but with 4 times sample points to obtain better precision
t=linspace(-4*pi,4*pi,257)
t=t[:-1] 
dt=t[1]-t[0]
fmax=1/dt
n=arange(256) 
wnd=fftshift(0.54+0.46*cos(2*pi*n/255)) 
y=sin(sqrt(2)*t)
y=y*wnd
y[0]=0 
y=fftshift(y)
Y=fftshift(fft(y))/256.0 
w=linspace(-pi*fmax,pi*fmax,257)
w=w[:-1]
figure(7)
subplot(2,1,1)
plot(w,abs(Y),'b',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$", size=16)
title(r"Spectrum of $\sin(\sqrt{2}t)\times w(t)$") 
grid(True)
subplot(2,1,2)
plot(w, angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"$\angle$$Y$", size=16)
xlabel(r"$\omega$", size=16) 
grid(True)
show()


'''
Q2
'''
# Spectrum of cos^3(0.86t) without hamming window
t=linspace(-pi,pi,65)
t=t[:-1]
dt=t[1]-t[0]
fmax=1/dt
y=cos(0.86*t)**3
y=fftshift(y) 
Y=fftshift(fft(y))/64.0 
w=linspace(-pi*fmax,pi*fmax,65)
w=w[:-1]
figure(1)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos^3(0.86t)$") 
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-10,10])
ylabel(r"$\angle$$Y$",size=16) 
xlabel(r"$\omega$",size=16)
grid(True)
show()

# Spectrum of cos^3(0.86t) with hamming window
t=linspace(-4*pi,4*pi,257)
t=t[:-1] 
dt=t[1]-t[0]
fmax=1/dt
n=arange(256) 
wnd=fftshift(0.54+0.46*cos(2*pi*n/255)) 
y=cos(0.86*t)**3
y=y*wnd
y=fftshift(y)
Y=fftshift(fft(y))/256.0 
w=linspace(-pi*fmax,pi*fmax,257)
w=w[:-1]
figure(2)
subplot(2,1,1)
plot(w,abs(Y),'b',lw=2)
xlim([-4,4])
ylabel(r"$|Y|$", size=16)
title(r"Spectrum of $\cos^3(0.86t)\times w(t)$") 
grid(True)
subplot(2,1,2)
plot(w, angle(Y),'ro',lw=2)
xlim([-4,4])
ylabel(r"$\angle$$Y$", size=16)
xlabel(r"$\omega$", size=16) 
grid(True)
show()


'''
Q3
'''
t=linspace(-pi,pi,129)
t=t[:-1] 
dt=t[1]-t[0]
fmax=1/dt
n=arange(128) 
wnd=fftshift(0.54+0.46*cos(2*pi*n/127)) 
wo = 1.35
delta = 0.8
y=cos(wo*t+delta)
y=y*wnd
y=fftshift(y) 
Y=fftshift(fft(y))/128.0 
w=linspace(-pi*fmax,pi*fmax,129)
w=w[:-1]
figure(1)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-8,8])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos(\omega t+\delta)\times w(t)$") 
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-8,8])
ylabel(r"$\angle$$Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()
# Code for estimating omega and delta
wEstimate = sum(abs(Y)**1.75 * abs(w))/sum(abs(Y)**1.75)
c1 = cos(wEstimate*t)
c2 = sin(wEstimate*t)
A = np.c_[c1, c2]
vals = lstsq(A, y)[0]
dEstimate = arctan2(-vals[1],vals[0])
print("wo Estimate: {:.03f} \t wo Actual value: {:.03f}".format(wEstimate, wo))
print("delta Estimate {:.03f} \t delta Actual value: {:.03f}".format(dEstimate, delta))

'''
Q4
Adding white gaussian noise to the above plots
'''
noise = 0.1*np.random.randn(128)
t=linspace(-pi,pi,129)
t=t[:-1] 
dt=t[1]-t[0]
fmax=1/dt
n=arange(128) 
wnd=fftshift(0.54+0.46*cos(2*pi*n/127)) 
wo = 1.35
delta = 0.8
y=cos(wo*t+delta)+noise
y=y*wnd
y=fftshift(y) 
Y=fftshift(fft(y))/128.0 
w=linspace(-pi*fmax,pi*fmax,129)
w=w[:-1]
figure(2)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-8,8])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos(\omega t+\delta)+noise \times w(t)$") 
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-8,8])
ylabel(r"$\angle$$Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()
wEstimate = sum(abs(Y)**1.75 * abs(w))/sum(abs(Y)**1.75)
c1 = cos(wEstimate*t)
c2 = sin(wEstimate*t)
A = np.c_[c1, c2]
vals = lstsq(A, y)[0]
dEstimate = np.arctan2(-vals[1], vals[0])
print("wo Estimate: {:.03f} \t wo Actual value: {:.03f}".format(wEstimate, wo))
print("delta Estimate {:.03f} \t delta Actual value: {:.03f}".format(dEstimate, delta))


'''
Q5
DFT of a chirped signal
'''
t=linspace(-pi,pi,1025)
t=t[:-1] 
dt=t[1]-t[0]
fmax=1/dt
n=arange(1024) 
wnd=fftshift(0.54+0.46*cos(2*pi*n/1023)) 
y=cos(16*(1.5*t + (t**2)/(2*pi)))
y=y*wnd
y=fftshift(y) 
Y=fftshift(fft(y))/1024.0 
w=linspace(-pi*fmax,pi*fmax,1025)
w=w[:-1]
figure(1)
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-80,80])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\cos(16(1.5t+(t^2)/(2\pi)))\times w(t)$") 
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-80,80])
ylabel(r"$\angle$$Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

'''
Q6
'''
# First calculating the DFT of x taking every batch size sample
x = cos(16*(1.5*t + (t**2)/(2*pi)))
t_batch = split(t, 1024//64)
x_batch = split(x, 1024//64)
X = np.zeros((1024//64, 64), dtype=complex)
for i in range(1024//64):
    X[i] = fftshift(fft(x_batch[i]))/64
#Plotting the 3D PLOT without multiplying by the hamming window
t = t[::64]
w = linspace(-fmax*pi,fmax*pi,65)[:-1]
t, w = meshgrid(t, w)

fig = plt.figure(1)
ax = fig.add_subplot(211, projection='3d')
surf = ax.plot_surface(w, t, abs(X).T, cmap='viridis')
colorbar(surf)
xlabel(r"Frequency $\to$")
ylabel(r"Time $\to$")
title(r"Magnitude $\|Y\|$")

ax = fig.add_subplot(212, projection='3d')
surf = ax.plot_surface(w, t, np.angle(X).T, cmap='viridis')
fig.colorbar(surf)
xlabel(r"Frequency $\to$")
ylabel(r"Time $\to$")
title(r"Angle $\angle Y$")
show()

#Plotting the 3D PLOT with multiplying by the hamming window
t = linspace(-pi, pi, 1025)[:-1]
x = cos(16*(1.5*t + (t**2)/(2*pi))) * wnd
t_batch = split(t, 1024//64)
x_batch = split(x, 1024//64)
X = np.zeros((1024//64, 64), dtype=complex)
for i in range(1024//64):
    X[i] = fftshift(fft(x_batch[i]))/64
t = t[::64]
w = linspace(-fmax*pi,fmax*pi,65)[:-1]
t, w = meshgrid(t, w)

fig = plt.figure(1)
ax = fig.add_subplot(211, projection='3d')
surf = ax.plot_surface(w, t, abs(X).T, cmap='viridis')
colorbar(surf)
xlabel(r"Frequency $\to$")
ylabel(r"Time $\to$")
title(r"Magnitude $\|Y\|$")

ax = fig.add_subplot(212, projection='3d')
surf = ax.plot_surface(w, t, np.angle(X).T, cmap='viridis')
fig.colorbar(surf)
xlabel(r"Frequency $\to$")
ylabel(r"Time $\to$")
title(r"Angle $\angle Y$")
show()