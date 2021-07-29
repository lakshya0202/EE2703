from pylab import * 

# Printing Fourier and Inverse Fourier Transform for a random function
x=rand(100) 
X=fft(x)
y=ifft(X)
c_[x,y]
print(abs(x-y).max())

# fft of sin(5x)
x=linspace(0,2*pi,128) 
y=sin(5*x)
Y=fft(y)
figure()
subplot(2,1,1)
plot(abs(Y),lw=2) 
ylabel(r"$|Y|$",size=16) 
title(r"Spectrum of $\sin(5t)$") 
grid(True)
subplot(2,1,2) 
plot(unwrap(angle(Y)),lw=2) 
ylabel(r"$\angle$$Y$",size=16) 
xlabel(r"$k$",size=16) 
grid(True)
show()

# Shifting the spectrum appropriately
x=linspace(0,2*pi,129)
x=x[:-1]
y=sin(5*x) 
Y=fftshift(fft(y))/128.0 
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10]) 
ylabel(r"$|Y|$",size=16) 
title(r"Spectrum of $\sin(5t)$") 
grid(True)
subplot(2,1,2) 
plot(w, angle(Y),'ro',lw=2) 
ii=where(abs(Y)>1e-3) 
plot(w[ii],angle(Y[ii]),'go',lw=2) 
xlim([-10,10])
ylabel(r"$\angle$$Y$",size=16) 
xlabel(r"$k$",size=16)
grid(True)
show()

# AM Modulation
t=linspace(0,2*pi,129)
t=t[:-1] 
y=(1+0.1*cos(t))*cos(10*t) 
Y=fftshift(fft(y))/128.0 
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $(1+0.1\cos(t))\cos(10t)$") 
grid(True)
subplot(2,1,2) 
plot(w,angle(Y),'ro',lw=2) 
xlim([-15,15])
ylabel(r"$\angle$$Y$",size=16) 
xlabel(r"$\omega$",size=16) 
grid(True)
show()

# AM Modulation with more sampling points
t=linspace(-4*pi,4*pi,513)
t=t[:-1] 
y=(1+0.1*cos(t))*cos(10*t) 
Y=fftshift(fft(y))/512.0 
w=linspace(-64,64,513)
w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-15,15])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$") 
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'ro',lw=2)
xlim([-15,15])
ylabel(r"$\angle$$Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()

# Spectrum of sin^3 t
x=linspace(0,2*pi,129)
x=x[:-1]
y=(sin(x))**3 
Y=fftshift(fft(y))/128.0 
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10]) 
ylabel(r"$|Y|$",size=16) 
title(r"Spectrum of $\sin^3(t)$") 
grid(True)
subplot(2,1,2) 
plot(w, angle(Y),'ro',lw=2) 
ii=where(abs(Y)>1e-3) 
plot(w[ii],angle(Y[ii]),'go',lw=2) 
xlim([-10,10])
ylabel(r"$\angle$$Y$",size=16) 
xlabel(r"$k$",size=16)
grid(True)
show()

# Spectrum of sin^3 t
x=linspace(0,2*pi,129)
x=x[:-1]
y=(cos(x))**3 
Y=fftshift(fft(y))/128.0 
w=linspace(-64,63,128)
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-10,10]) 
ylabel(r"$|Y|$",size=16) 
title(r"Spectrum of $\cos^3(t)$") 
grid(True)
subplot(2,1,2) 
plot(w, angle(Y),'ro',lw=2) 
ii=where(abs(Y)>1e-3) 
plot(w[ii],angle(Y[ii]),'go',lw=2) 
xlim([-10,10])
ylabel(r"$\angle$$Y$",size=16) 
xlabel(r"$k$",size=16)
grid(True)
show()

# Spectrum of cos(20t+5cos(t))
t=linspace(-4*pi,4*pi,513)
t=t[:-1] 
y=cos(20*t+5*cos(t)) 
Y=fftshift(fft(y))/512.0 
w=linspace(-64,64,513)
w=w[:-1]
figure()
subplot(2,1,1)
plot(w,abs(Y),lw=2)
xlim([-40,40])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $cos(20t+5cos(t))$") 
grid(True)
subplot(2,1,2)
#plot(w,angle(Y),'ro',lw=2)
# Plotting points which are above a certain magnitude
ii=where(abs(Y)>1e-3) 
plot(w[ii],angle(Y[ii]),'go',lw=2) 
xlim([-40,40])
ylabel(r"$\angle$$Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()


# Spectrum of Gaussian exponent
t=linspace(-8*pi,8*pi,513)
t=t[:-1] 
y=exp(-(t**2)/2) 
Y=fftshift(fft(y))*5/256
w=linspace(-64,64,513)
w=w[:-1]
Y_ = sqrt(2*pi)*exp(-(w**2)/8)/5
print(max(abs(abs(Y)-Y_)))
figure()
subplot(2,1,1)
plot(w,abs(Y),'b')
plot(w,abs(Y_),'ro')
xlim([-10,10])
ylabel(r"$|Y|$",size=16)
title(r"Spectrum of $exp(\frac{-t^2}{2})$") 
grid(True)
subplot(2,1,2)
plot(w,angle(Y),'bo',lw=2) 
xlim([-10,10])
ylabel(r"$\angle$$Y$",size=16)
xlabel(r"$\omega$",size=16)
grid(True)
show()


