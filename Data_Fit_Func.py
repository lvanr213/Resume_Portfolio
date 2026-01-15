import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data_filename = 'open_port.csv' 

with open(data_filename ,"r") as f:
    for x,line in enumerate(f):
        if(line == 'Channel Data:\n'): 
            data_start = x+2
            print("Data starts on row {}".format(data_start))
            break

data = np.loadtxt('open_port.csv' , unpack=True, skiprows = 29, delimiter= ",",usecols=[0,2])


channel = data[0]
if(len(data)==2): 
    N = data[1]
else:
    Energy = data[1]
    N = data[2]

dN = np.sqrt(N)
for i, value in enumerate(dN):
    if value == 0:
        dN[i] = 1.14

min_value = 460
max_value = 520
count_estimate = 150000
center_estimate = 490
width_estimate = 20
slope = 0
offset = 0

fig,ax = plt.subplots()
ax.errorbar(channel, N, dN, fmt='k.',alpha = 0.5,label='Data')
p02 = [count_estimate, center_estimate , width_estimate,slope,offset]
channel_cont = np.linspace(min_value, max_value , 5000)
#ax.plot(channel_cont, gaussianlinear(p02, channel_cont), 'r--', label='Initial',zorder=10)
#ax.axvspan(min_value ,max_value,label='Fit Region',alpha = 0.5)


ax.set_xlabel('Channels')
ax.set_ylabel('Counts')
ax.set_title("Open Port 15cm Lead Energy Spectrum")
ax.legend()

#plt.savefig('open_port_15_lead.png',dpi=300)


data = np.loadtxt(data_filename , unpack=True, skiprows = 29, delimiter= ",",usecols=[0,2])

channel = data[0]
if(len(data)==2): 
    N = data[1]
else:
    Energy = data[1]
    N = data[2]

dN = np.sqrt(N)
for i, value in enumerate(dN):
    if value == 0:
        dN[i] = 1.14
def gaussianfunc(p,x):
    return p[0]/(p[2]*np.sqrt(2*np.pi))*np.exp(-(x-p[1])**2/(2*p[2]**2))

def linearfunc(p,x):
    return p[0]*x + p[1]

def gaussianlinear(p,x):
    return gaussianfunc(p[0:3],x) + linearfunc(p[3:5],x)

def residual(p,func, xvar, yvar, err):
    return (func(p, xvar) - yvar)/err
# The code below defines a data fitting function.
# Inputs are:
# initial guess for parameters p0
# the function we're fitting to
# the x,y, and dy variables
# tmi can be set to 1 or 2 if more intermediate data is needed

def data_fit(p0, func, xvar, yvar, err, tmi=0):
    try:
        fit = optimize.least_squares(residual, p0, args=(func,xvar, yvar, err), verbose=tmi)
    except Exception as error:
        print("Something has gone wrong:",error)
        return p0, np.zeros_like(p0), np.nan, np.nan
    pf = fit['x']

    print()

    try:
        cov = np.linalg.inv(fit['jac'].T.dot(fit['jac']))          
        # This computes a covariance matrix by finding the inverse of the Jacobian times its transpose
        # We need this to find the uncertainty in our fit parameters
    except:
        # If the fit failed, print the reason
        print('Fit did not converge')
        print('Result is likely a local minimum')
        print('Try changing initial values')
        print('Status code:', fit['status'])
        print(fit['message'])
        return pf, np.zeros_like(pf), np.nan, np.nan
            

    chisq = sum(residual(pf, func, xvar, yvar, err) **2)
    dof = len(xvar) - len(pf)
    red_chisq = chisq/dof
    pferr = np.sqrt(np.diagonal(cov)) 
    print('Converged with chi-squared {:.2f}'.format(chisq))
    print('Number of degrees of freedom, dof = {:.2f}'.format(dof))
    print('Reduced chi-squared {:.2f}'.format(red_chisq))
    print()
    Columns = ["Parameter #","Initial guess values:", "Best fit values:", "Uncertainties in the best fit values:"]
    print('{:<11}'.format(Columns[0]),'|','{:<24}'.format(Columns[1]),"|",'{:<24}'.format(Columns[2]),"|",'{:<24}'.format(Columns[3]))
    for num in range(len(pf)):
        print('{:<11}'.format(num),'|','{:<24.3e}'.format(p0[num]),'|','{:<24.3e}'.format(pf[num]),'|','{:<24.3e}'.format(pferr[num]))
    return pf, pferr, chisq,dof

fig,ax = plt.subplots()
ax.errorbar(channel, N, dN, fmt = 'k.',alpha = 0.5)
ax.set_xlim(min_value-50,max_value+50)
ax.set_ylim(0,300)

channel2 = channel[min_value:max_value]
N2 = N[min_value:max_value]
dN2 = dN[min_value:max_value]

p0 = [count_estimate, center_estimate , width_estimate]
channel_cont = np.linspace(min(channel2), max(channel2), 5000)
ax.plot(channel_cont, gaussianfunc(p0, channel_cont), 'r--', label='Initial',zorder=10)

print("Gaussian only")
pf1, pferr1, chisq1, dof1 = data_fit(p0, gaussianfunc, channel2, N2, dN2)
ax.plot(channel_cont, gaussianfunc(pf1, channel_cont), 'b-', label='No Background',zorder=11)

print("\n\nGaussian with linear background")
p02 = [count_estimate, center_estimate , width_estimate,slope,offset]
pf2, pferr2, chisq2, dof2 = data_fit(p02, gaussianlinear, channel2, N2, dN2)
ax.plot(channel_cont, gaussianlinear(pf2, channel_cont), 'g-', label='With Background',zorder=11)

textfit = '$R(x) = Ae^(-𝜆x) + B $ \n'  
textfit += '$A = {:.2f} \pm {:.2f} \ counts/s $ \n'.format(pf2[0],pf2[0])
textfit += '$B = {:.2f} \pm {:.2f} \ counts/s $ \n'.format(pf2[4],pf2[4])
textfit +='$𝜆 = {:.4f} \pm {:.4f}  \ mm^(-1) $ \n'.format(pf2[1],pf2[1])
textfit += '$\chi^2= {:.2f}$ \n'.format(chisq2) 
textfit += '$N = {}$ (dof) \n'.format(dof2) 
textfit += '$\chi^2/N = {:.2f}$'.format(chisq2/dof2) 
ax.text(0.01, 0.99, textfit, transform=ax.transAxes , fontsize=8,verticalalignment='top')
ax.set_xlabel('Channels')
ax.set_ylabel('Counts')
ax.set_title("15cm Lead Energy Peak")
ax.legend()

#plt.savefig('open_port_15_lead_peak.png',dpi=300)