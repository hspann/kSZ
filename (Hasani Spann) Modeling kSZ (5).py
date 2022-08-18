#!/usr/bin/env python
# coding: utf-8

# In[243]:


get_ipython().run_line_magic('matplotlib', 'inline')
#%load_ext autoreload
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # For mac users with Retina display")

import sys
sys.path.append('../../theorymodel/theorymodel/')
import pyccl as ccl  
#import tracer
#from base import *  
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.integrate import simpson
from ksz import *
import sympy
from sympy import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


# # Set Up Cosmology Class

# In[195]:


cosmo = ccl.Cosmology(
        Omega_b=0.048206,
        Omega_c=0.258909,
        h=0.6777,
        n_s=0.9611,
        A_s=2.0983651e-9,
        Neff=3.046)
   
l     = np.arange(6001)
k_arr = np.geomspace(1e-4,1e2,256)
a_arr = np.linspace(0.1,1,100)
r_arr = np.geomspace(1E-2,1E2,256)
x_arr = np.geomspace(1e-4,30,256)


# # Set Up kSZ Class

# In[196]:


mykSZ = kSZ(cosmo)


# # Integrand of the kSZ power spectrum for different multipoles

# In[197]:


get_ipython().run_line_magic('time', 'integ = mykSZ.ksz_integrand(mykSZ.zs, 5000)')


# In[198]:


ls = [100, 1000, 3000, 5000]
for l in ls:
    plt.plot(mykSZ.zs, mykSZ.ksz_integrand(mykSZ.zs, l), label=r'$\ell=%d$'%l)
plt.grid()
plt.xlabel(r'$z$',size=15)
plt.ylabel(r'$\partial C_{\ell}^{\rm kSZ}/\partial z$',size=15)
plt.legend()


# In[199]:


ls = mykSZ.ls
plt.plot(ls, ls*(ls+1.) * kSZ(cosmo, linear=True).get_cl() / (2*np.pi), label=r'Ostriker-Vishniac')
plt.plot(ls, ls*(ls+1.) * kSZ(cosmo, linear=False).get_cl() / (2*np.pi), label=r'Full kSZ')
plt.plot(ls, ls*(ls+1.) * kSZ(cosmo, linear=False, baryonic=True).get_cl() / (2*np.pi), label=r'Full kSZ (w/ gas window function)')
plt.xlabel(r'$\ell$', size=15)
plt.ylabel(r'$\mathcal{D}_{\ell}^{\rm kSZ}$ [$\mu$K$^2$]', size=15)
plt.grid()
plt.legend(loc='best')


# # $C^{kSZ}_{l}$ as function of the He ionization state (assumed constant in redshift)

# In[200]:


NHes = [2,1,0]
ls = mykSZ.ls
for NHe in NHes:
    mykSZ = kSZ(cosmo, NHe=NHe)
    plt.plot(ls, ls*(ls+1.) * mykSZ.get_cl() / (2*np.pi), label=r'$N_{\rm He}=%d$'%NHe)
plt.xlabel(r'$\ell$', size=15)
plt.ylabel(r'$\mathcal{D}_{\ell}^{\rm kSZ}$ [$\mu$K$^2$]', size=15)
plt.legend()
plt.grid()


# # (New Content)

# ### Taking The late-Time kSZ Spectrum

# In[201]:


ls = mykSZ.ls
plt.plot(ls, mykSZ.get_cl(ls)*ls*(ls+1)/(2*np.pi), label=r'Late-Time kSZ')
plt.xlabel(r'$\ell$', size=15)
plt.ylabel(r'$\mathcal{C}_{\ell}^{\rm kSZ}$ [$\mu$K$^2$]', size=15)
plt.grid()
plt.legend(loc='best')
#(Convention) D_ell is ell*(ell+1)*C_ell/(2*pi)


# ### Interpolating Late-Time kSZ

# In[202]:


plt.plot(ls, mykSZ.get_cl(ls)*ls*(ls+1)/(2*np.pi), label=r'Late-Time kSZ')
new_l_min = 2
new_l_max = 4000
new_l_diff = new_l_max - new_l_min
new_l = np.arange(new_l_min,new_l_max) # new ell values, for example
dlki= np.interp(new_l, ls, mykSZ.get_cl(ls)*ls*(ls+1)/(2*np.pi))
print(dlki)
plt.plot(new_l, dlki, label=r'Interpolated Late-Time kSZ')
plt.xlabel(r'$\ell$', size=15)
plt.ylabel(r'$\mathcal{C}_{\ell}^{\rm kSZ}$', size=15)
plt.grid()
plt.legend(loc='best')


# In[203]:


print(mykSZ.get_cl(ls))
print(mykSZ.get_cl(ls)*ls*(ls+1)/(2*np.pi))
print(dlki)


# In[ ]:





# ### Interpolating kSZ Battaglia

# In[204]:


import numpy as np
from scipy.interpolate import interp1d

l, Dl_kSZ_battaglia = np.loadtxt('Dl_ksz_reion_SF.csv', delimiter=',', unpack = True)
# new ell values, for example

# numpy approach
Dl_kSZ_battaglia_interpolated_numpy = np.interp(new_l, l, Dl_kSZ_battaglia)*new_l*(new_l+1)/(2*np.pi)
print(Dl_kSZ_battaglia_interpolated_numpy)
plt.plot(new_l, Dl_kSZ_battaglia_interpolated_numpy, label=r'Battaglia kSZ')
plt.xlabel(r'$\ell$', size=15)
plt.ylabel(r'$\mathcal{C}_{\ell}^{\rm Bat}$', size=15)
plt.grid()
plt.legend(loc='best')

#scipy approach
#Dl_kSZ_battaglia_interpolated_scipy = interp1d(l, Dl_kSZ_battaglia) # this is a function you can actually call!


# ### Using C_Battaglia to get reionization kSZ

# In[205]:


def ion_kSZ_integrand(z, kSZ_Bat, z_rei, sigma_rei):
    #z and sigma are constants, kSZ_Bat is the kSZ from Battaglia
    return ((np.exp((-(z-z_rei)**2)/(2*sigma_rei**2))/((2*np.pi*sigma_rei**2)**(1/2)))*kSZ_Bat)

z_rei = 8.8
sigma_rei = 1.0
z_min = 0
z_max = 20
ion_kSZ_intermediate = []
ion_kSZ = []

for i in range(np.size(Dl_kSZ_battaglia_interpolated_numpy)):
    kSZ_Bat = Dl_kSZ_battaglia_interpolated_numpy[i]
    ion_kSZ_intermediate.append(quad(ion_kSZ_integrand, z_min, z_max, args=(kSZ_Bat,z_rei,sigma_rei)))
    ion_kSZ.append(ion_kSZ_intermediate[i][1])
    
#Parameters are the Battaglia kSZ function, minimum redshift z, maximum redshift z_max, and constants z and sigma

ion_kSZ_scaled = ion_kSZ*new_l*(new_l+1)/(2*np.pi)

plt.plot(new_l, ion_kSZ_scaled, label=r'Reionization kSZ')
plt.xlabel(r'$\ell$', size=15)
plt.ylabel(r'$\mathcal{C}_{\ell}^{\rm kSZ}$ [$\mu$K$^2$]', size=15)
plt.grid()
plt.legend(loc='best')


# In[237]:


print(len(Dl_kSZ_battaglia_interpolated_numpy))


# In[206]:


#testing the integration function
def integrand(x, a, b):
    return a*x**2 + b

a = 2
b = 1
I = quad(integrand, 0, 1, args=(a,b))
print(I)


# ### Adding Late-time and Ionizatgion kSZ

# In[207]:


full_kSZ = []

for i in range(len(ion_kSZ)):
    full_kSZ.append(ion_kSZ[i]+dlki[i])

print(full_kSZ[0])
print(dlki[0])
print(ion_kSZ[0])


# ## Obtaining C^tot from CAMB

# In[208]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(4000, lens_potential_accuracy=0);


# In[209]:


#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(new_l_max, lens_potential_accuracy=0);


# In[210]:


#calculate results for these parameters
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)


# In[211]:


#plot the total lensed CMB power spectra
totCL=powers['total']
print(totCL.shape)
#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
ells = np.arange(totCL.shape[0])
fig, ax = plt.subplots(2,2, figsize = (12,12))
ax[0,0].plot(ells, totCL[:,0], color= 'k')
ax[0,0].set_title(r'Tot Lensed CMB Angular Power Spectrum (muK Units)')
#ax[0,1].plot(ells, mykSZ.get_cl(ells), color= 'r');
#ax[0,1].set_title(r'General kSZ Spectrum (muK^2 Units)')
#ax[1,0].plot(ls, ((totCL[:,0])**(1/2))/(mykSZ.get_cl(ells)), color='k')
#ax[1,0].set_title(r'Tot CMB / kSZ Spectrum')
#ax[1,1].plot(ls, ((mykSZ.get_cl(ells))**(1/2))/(totCL[:,0]), color='k');
#ax[1,0].set_title(r'kSZ Spectrum / Tot CMB')

for ax in ax.reshape(-1): ax.set_xlim([2,4000]);

#interpolate the power spectra:
totCL_interp=np.interp(new_l, ells, totCL[:,0])
print(totCL_interp)


# ### Obtaining Ws_l from C_tot and C_kSZ

# In[212]:


Ws = []

for i in range(len(full_kSZ)):
    Ws.append((full_kSZ[i])**(1/2)/totCL_interp[i])

print(len(Ws))


# ### Obtaining K_bar

# In[213]:


def K_bar_integrand(W, C_tot):
    K_bar_integrand = []
    for i in range(len(W)):
        K_bar_integrand.append(C_tot[i]*(W[i]**2)*new_l[i]/(2*np.pi))
    return K_bar_integrand

K_bar_int = K_bar_integrand(Ws, totCL_interp)

K_bar = np.trapz(K_bar_int)
print (K_bar)


# ## Obtaining late time kSZ integrand for dK_bar (along l and z)

# In[214]:


#Full list, excessive time

#late_time_kSZ_integrand = []
#Z = []

#for i in range(1, 1101):
#     Z.append(i)

#for l_int in range (new_l_diff):
#    late_time_kSZ_integrand.append(mykSZ.ksz_integrand(Z,l_int))new


# In[215]:


late_time_kSZ_integrand = []
z_lt_max = 6
Z_lt = np.linspace(0.000001, z_lt_max, 50)
print(Z_lt)

for l_int in range (new_l_min, new_l_max, 290):
    late_time_kSZ_integrand.append(mykSZ.ksz_integrand(Z_lt,l_int))
    
l_int = np.arange(new_l_min, new_l_max, 290)


# In[216]:


print((late_time_kSZ_integrand))
print(l_int)
print(late_time_kSZ_integrand[0][1])


# In[217]:


late_time_kSZ_integrand_zx_ly = []
for i in range(len(late_time_kSZ_integrand[0])):
    l_array = []
    for k in range(len(late_time_kSZ_integrand)):
        l_array.append(late_time_kSZ_integrand[k][i])
    late_time_kSZ_integrand_zx_ly.append(l_array)

print(late_time_kSZ_integrand_zx_ly)


# In[218]:


ltki_interp = []
for i in range (len(late_time_kSZ_integrand_zx_ly)):
    ltki_interp.append(np.interp(new_l, l_int, late_time_kSZ_integrand_zx_ly[i]))

print(len(ltki_interp))
print(len(ltki_interp[0]))

#Z_lt_max = 6
#Edit here to set upper redshift value. 
#251 is where python begins to interpret the values as infinity. 
#50 gives decipherable plots. Figured by trial and error
#6 one apporximate estimation of the end of reionization. The upper duration limit was about 2.4, giving range z ~ 6-8.4
#ltki_cutoff = np.resize(ltki_interp,(Z_lt_max+1,new_l_diff))
#print(len(ltki_cutoff))
#print(len(ltki_cutoff[0]))
#print(ltki_cutoff)


# In[219]:


#3D plotting (failed):

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
# your z_cutoff range is defined higher up
# so is new_l

#l_ltki = new_l
#Z_ltki = np.arange(0, Z_cutoff)

#X, Y = np.meshgrid(l_ltki, Z_ltki)

# Plot the surface.
#surf = ax.plot_surface(X, Y, a, cmap=cm.coolwarm,
 #                      linewidth=0, antialiased=False)

# Customize the ltki axis.
#ax.set_zlim(5.56268065e+55, 1.77837953e+308)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=1e+100, aspect=5)

#plt.show()


# In[220]:


#Contour plotting (successful?)

l_ltki = new_l

X, Y = np.meshgrid(l_ltki, Z_lt)

plt.title("Late-Time kSZ Integrand--Revised")
plt.xlabel("l")
plt.ylabel("z")
plt.contourf(X, Y, ltki_interp, 100, cmap='RdGy')
plt.colorbar();


# In[221]:


ltki_interp = np.asarray(ltki_interp)
ltki_interp = ltki_interp*mykSZ.T0_square * mykSZ.const_kSZ
Cl_kSZ_late = []
for i in range(ltki_interp.shape[1]):
    Cl_kSZ_late.append(np.trapz(ltki_interp[:,i], x = Z_lt))

Cl_kSZ_late = np.asarray(Cl_kSZ_late) *new_l*(new_l+1)/(2*np.pi)*mykSZ.T0_square * mykSZ.const_kSZ / (2*new_l+1) ** 3
print(Cl_kSZ_late.size)


#use scipy.integrate simps


# In[222]:


print(Cl_kSZ_late)
len(ltki_interp[0])


# In[223]:


plt.plot(ls, mykSZ.get_cl(ls)*ls*(ls+1)/(2*np.pi), label=r'Late-Time kSZ')
A= Cl_kSZ_late
plt.plot(new_l, A, label=r'Late-Time kSZ (from integrand)')
plt.xlabel(r'$\ell$', size=15)
plt.ylabel(r'$\mathcal{C}_{\ell}^{\rm kSZ}$ [$\mu$K$^2$]', size=15)
plt.grid()
plt.legend(loc='best')
#(Convention) D_ell is ell*(ell+1)*C_ell/(2*pi)
# Note: ell is always 


# ### Obtaining Reionization kSZ

# In[224]:


def ion_kSZ_integrand(z, kSZ_Bat, z_rei, sigma_rei):
    #z_rei and sigma_rei are constants, kSZ_Bat is the kSZ from Battaglia
    return ((np.exp((-(z-z_rei)**2)/(2*sigma_rei**2))/((2*np.pi*sigma_rei**2)**(1/2)))*kSZ_Bat)


z_rei = 8.8
sigma_rei = 1.0
ion_kSZ_integrand_arr = []
kSZ_Bat = Dl_kSZ_battaglia_interpolated_numpy
Z_rei_max = 20
Z_rei_min = 6
Z_rei = np.linspace(Z_rei_min, Z_rei_max, 50)



for i in range(len(Z_rei)):
    ion_kSZ_integrand_arr.append(ion_kSZ_integrand(Z_rei[i], kSZ_Bat, z_rei, sigma_rei))

print(len(ion_kSZ_integrand_arr[0]))
print((ion_kSZ_integrand_arr))


# ### Obtaining Full kSZ integrand

# In[225]:


full_kSZ_integrand_append = np.concatenate((ltki_interp, ion_kSZ_integrand_arr))
Z_full_append = np.concatenate((Z_lt, Z_rei))
#print ((full_kSZ_integrand_append))
#print ((Z_full_append))
#print ((ltki_interp))
#print ((ion_kSZ_integrand_arr))
#note: there's overlap between the two arrays at z = z_lt_max = z_rei_min, and indexing is different between before and after that value


# ### Obtaining dK_bar/dz

# In[226]:


#Using full (lt and rei). Note: This is currently janky
dC_kSZ_dz_full = full_kSZ_integrand_append
dK_bar_integrand_full = []
for i in range(len(dC_kSZ_dz_full)):
    dK_bar_integrand_intermediate = []
    for k in range (2, len(new_l)):
    #note, if involving the ell = 0 index, we get all nan output in dK_bar_dz, when involving ell = 1 index, we get all z = inf output in d_K_bar_dz 
        dK_bar_integrand_intermediate.append(dC_kSZ_dz_full[i][k]*(Ws[k]**2)*new_l[k]/(2*np.pi))
    dK_bar_integrand_full.append(dK_bar_integrand_intermediate)

dK_bar_dz_full = []

for i in (range(len(dK_bar_integrand_full))):
    dK_bar_dz_full.append(np.trapz(dK_bar_integrand_full[i]))

    

#For lt only
dC_kSZ_dz_lt= ltki_interp
dK_bar_integrand_lt = []
for i in range(len(Z_lt)):
    dK_bar_integrand_intermediate = []
    for k in range(2, len(new_l)):
    #note, if involving the ell = 0 index, we get all nan output in dK_bar_dz, when involving ell = 1 index, we get all z = inf output in d_K_bar_dz 
        dK_bar_integrand_intermediate.append(dC_kSZ_dz_lt[i][k]*(Ws[k]**2)*new_l[k]/(2*np.pi))
    dK_bar_integrand_lt.append(dK_bar_integrand_intermediate)

dK_bar_dz_lt = []

for i in (range(len(dK_bar_integrand_lt))):
    dK_bar_dz_lt.append(np.trapz(dK_bar_integrand_lt[i]))
    

    
    
#For rei only
dC_kSZ_dz_rei= ion_kSZ_integrand_arr
dK_bar_integrand_rei = []
for i in range(len(Z_rei)):
    dK_bar_integrand_intermediate = []
    for k in range(2, len(new_l)):
    #note, if involving the ell = 0 index, we get all nan output in dK_bar_dz, when involving ell = 1 index, we get all z = inf output in d_K_bar_dz 
        dK_bar_integrand_intermediate.append(dC_kSZ_dz_rei[i][k]*(Ws[k]**2)*new_l[k]/(2*np.pi))
    dK_bar_integrand_rei.append(dK_bar_integrand_intermediate)

dK_bar_dz_rei = []

for i in (range(len(dK_bar_integrand_rei))):
    dK_bar_dz_rei.append(np.trapz(dK_bar_integrand_rei[i]))

#print(dK_bar_dz_full)
#print(dK_bar_dz_lt)
#print(dK_bar_dz_rei)


# In[227]:


plt.plot(Z_lt, dK_bar_dz_lt, label=r'Late-Time $\mathcal{K}_{bar}$/$dZ$')
A= Cl_kSZ_late
plt.xlabel(r'Z', size=15)
plt.ylabel(r'$\mathcal{K}_{bar}$/$dZ$', size=15)
plt.grid()
plt.legend(loc='best')


# In[228]:


plt.plot(Z_rei, dK_bar_dz_rei, label=r'Reionization $\mathcal{K}_{bar}$/$dZ$')
plt.xlabel(r'Z', size=15)
plt.ylabel(r'$\mathcal{K}_{bar}$/$dZ$', size=15)
plt.grid()
plt.legend(loc='best')


# In[229]:


plt.plot(Z_full_append, dK_bar_dz_full, label=r'"Full" $\mathcal{K}_{bar}$/$dZ$')
plt.xlabel(r'Z', size=15)
plt.ylabel(r'$\mathcal{K}_{bar}$/$dZ$', size=15)
plt.grid()
plt.legend(loc='best')


# ## P_n(k) function (Power Spectrum of field n at wavenumber k)

# ### Velocity Power Spectrum Pv(k) [For P_n(k)]

# In[230]:


def P_vel(k, z):
    #k is wavenumber, z is redshift, a is scale factor, f is structural growth rate
    a = 1/(1+z)
    f = ccl.growth_rate(cosmo, a)
    a_dot = a*ccl.h_over_h0(cosmo, a) * mykSZ.cosmo['H0']
    return((f*a_dot/(k))**2*ccl.linear_matter_power(cosmo, k, a))


# ### P_n(k) [Using Quad Integration] 

# In[231]:


def P_n_mu_int(mu, k_prime, k, z = 0):
    #-2*k*k_prime*mu subtraction is removed from the k^2 plus k_prime^2
    return(-(k_prime**2*mu**4)/(k**2+k_prime**2)*P_vel(k_prime, z)*P_vel(((k**2+k_prime**2)**(1/2)), z))

def P_n_k_prime_int(k_prime, k, z = 0):
    a = quad(P_n_mu_int, -1, 1, args = (k_prime, k, z))
    return k_prime**2*2*np.pi*(a[0])

def P_n(k, z, k_min = 1e-4, k_max = 4000):
    b = quad(P_n_k_prime_int, k_min, k_max, args = (k, z))
    return b


# In[232]:


test = P_n(1, 0)


# In[233]:


print(test)


# ### P_n(k) [Using Simpson's Integration]

# In[234]:


def P_n(k, z = 0, k_prime_min = 0.0001, k_prime_max = 4000, k_prime_index = 1, mu_min = -1, mu_max = 1, mu_index = 0.01):
    k_prime_interval = np.arange(k_prime_min, k_prime_max, k_prime_index)
    mu_interval = np.arange(mu_min, mu_max, mu_index)
    z = 0
    #creating arrays with the integrand data, and integrating the mu integrand to insert into k_prime integrand
    P_n_array_k_prime = []
    for i in (k_prime_interval):
        P_n_array_mu = []
        for j in mu_interval:
            P_n_array_mu.append(P_n_mu_int(j, i, k, z))
        P_n_array_k_prime.append(i**2*2*np.pi*simpson(P_n_array_mu, mu_interval))
    P_n_k = simpson(P_n_array_k_prime, k_prime_interval)
    return P_n_k    


# In[235]:


P_n(1)


# Note: The P_n(k) calculation is erroneous for unknown reasons atm. As such, we will return to C^KK_L later, if possible.

# In[240]:


print(new_l)


# # Calculating Noise Function N^KK_L

# In[245]:


N_KK_L = []

Ls = np.arange(0, 301)
print(Ls)
for i in Ls:
    N_integrand = []
    for j in new_l:
        N_integrand.append(j/(2*np.pi)*(Ws[j-2])**2*(Ws[i-(j-2)])**2*totCL_interp[j-2]*totCL_interp[i-(j-2)])
    N_KK_L.append(simpson(N_integrand, new_l))

print(N_KK_L)
        


# In[253]:


plt.plot(Ls, N_KK_L)
plt.title(r'Reconstruction Noise $\mathcal{N}^{KK}_{L}$')
plt.xlabel(r'L', size=15)
plt.ylabel(r'$\mathcal{K}^{−2}_{tot}{L}^2{C}^{KK}_{L}/{2π}$', size=15)
plt.grid()
plt.legend(loc='best')


# In[ ]:





# In[ ]:




