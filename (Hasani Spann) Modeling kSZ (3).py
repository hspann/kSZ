#!/usr/bin/env python
# coding: utf-8

# In[61]:


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
from ksz import *
import sympy
from sympy import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


# # Set Up Cosmology Class

# In[62]:


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

# In[63]:


mykSZ = kSZ(cosmo)


# # Integrand of the kSZ power spectrum for different multipoles

# In[64]:


get_ipython().run_line_magic('time', 'integ = mykSZ.ksz_integrand(mykSZ.zs, 5000)')


# In[65]:


ls = [100, 1000, 3000, 5000]
for l in ls:
    plt.plot(mykSZ.zs, mykSZ.ksz_integrand(mykSZ.zs, l), label=r'$\ell=%d$'%l)
plt.grid()
plt.xlabel(r'$z$',size=15)
plt.ylabel(r'$\partial C_{\ell}^{\rm kSZ}/\partial z$',size=15)
plt.legend()


# In[66]:


ls = mykSZ.ls
plt.plot(ls, ls*(ls+1.) * kSZ(cosmo, linear=True).get_cl() / (2*np.pi), label=r'Ostriker-Vishniac')
plt.plot(ls, ls*(ls+1.) * kSZ(cosmo, linear=False).get_cl() / (2*np.pi), label=r'Full kSZ')
plt.plot(ls, ls*(ls+1.) * kSZ(cosmo, linear=False, baryonic=True).get_cl() / (2*np.pi), label=r'Full kSZ (w/ gas window function)')
plt.xlabel(r'$\ell$', size=15)
plt.ylabel(r'$\mathcal{D}_{\ell}^{\rm kSZ}$ [$\mu$K$^2$]', size=15)
plt.grid()
plt.legend(loc='best')


# # $C^{kSZ}_{l}$ as function of the He ionization state (assumed constant in redshift)

# In[67]:


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

# In[68]:


ls = mykSZ.ls
plt.plot(ls, mykSZ.get_cl(ls)*ls*(ls+1)/(2*np.pi), label=r'Late-Time kSZ')
plt.xlabel(r'$\ell$', size=15)
plt.ylabel(r'$\mathcal{C}_{\ell}^{\rm kSZ}$ [$\mu$K$^2$]', size=15)
plt.grid()
plt.legend(loc='best')
#(Convention) D_ell is ell*(ell+1)*C_ell/(2*pi)


# ### Interpolating Late-Time kSZ

# In[69]:


plt.plot(ls, mykSZ.get_cl(ls)*ls*(ls+1)/(2*np.pi), label=r'Late-Time kSZ')
new_l_min = 0
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


# In[70]:


print(mykSZ.get_cl(ls))
print(mykSZ.get_cl(ls)*ls*(ls+1)/(2*np.pi))
print(dlki)


# In[ ]:





# ### Interpolating kSZ Battaglia

# In[71]:


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

# In[72]:


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


# In[73]:


#testing the integration function
def integrand(x, a, b):
    return a*x**2 + b

a = 2
b = 1
I = quad(integrand, 0, 1, args=(a,b))
print(I)


# ### Adding Late-time and Ionizatgion kSZ

# In[74]:


full_kSZ = []

for i in range(len(ion_kSZ)):
    full_kSZ.append(ion_kSZ[i]+dlki[i])

print(full_kSZ[0])
print(dlki[0])
print(ion_kSZ[0])


# ## Obtaining C^tot from CAMB

# In[75]:


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


# In[76]:


#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(new_l_max, lens_potential_accuracy=0);


# In[77]:


#calculate results for these parameters
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)


# In[78]:


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

# In[79]:


Ws = []

for i in range(len(full_kSZ)):
    Ws.append((full_kSZ[i])**(1/2)/totCL_interp[i])

print(len(Ws))


# ### Obtaining K_bar

# In[80]:


def K_bar_integrand(W, C_tot):
    K_bar_integrand = []
    for i in range(len(W)):
        K_bar_integrand.append(C_tot[i]*(W[i]**2)*new_l[i]/(2*np.pi))
    return K_bar_integrand

K_bar_int = K_bar_integrand(Ws, totCL_interp)

K_bar = np.trapz(K_bar_int)
print (K_bar)


# ## Obtaining late time kSZ integrand for dK_bar (along l and z)

# In[82]:


#Full list, excessive time

#late_time_kSZ_integrand = []
#Z = []

#for i in range(1, 1101):
#     Z.append(i)

#for l_int in range (new_l_diff):
#    late_time_kSZ_integrand.append(mykSZ.ksz_integrand(Z,l_int))new


# In[83]:


late_time_kSZ_integrand = []
Z = np.linspace(0.00001,50,51)

for l_int in range (new_l_min, new_l_max, 290):
    late_time_kSZ_integrand.append(mykSZ.ksz_integrand(Z,l_int))
    
l_int = np.arange(new_l_min, new_l_max, 290)


# In[84]:


print((late_time_kSZ_integrand))
print(l_int)
print(late_time_kSZ_integrand[0][1])


# In[85]:


late_time_kSZ_integrand_zx_ly = []
for i in range(len(late_time_kSZ_integrand[0])):
    l_array = []
    for k in range(len(late_time_kSZ_integrand)):
        l_array.append(late_time_kSZ_integrand[k][i])
    late_time_kSZ_integrand_zx_ly.append(l_array)

print(late_time_kSZ_integrand_zx_ly)


# In[128]:


ltki_interp = []
for i in range (len(late_time_kSZ_integrand_zx_ly)):
    ltki_interp.append(np.interp(new_l, l_int, late_time_kSZ_integrand_zx_ly[i]))

print(len(ltki_interp))
print(len(ltki_interp[0]))

Z_lt_max = 6
#Edit here to set upper redshift value. 
#251 is where python begins to interpret the values as infinity. 
#50 gives decipherable plots. Figured by trial and error
#6 one apporximate estimation of the end of reionization. The upper duration limit was about 2.4, giving range z ~ 6-8.4
ltki_cutoff = np.resize(ltki_interp,(Z_lt_max+1,new_l_diff))
print(len(ltki_cutoff))
print(len(ltki_cutoff[0]))
print(ltki_cutoff)


# In[129]:


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


# In[132]:


#Contour plotting (successful?)

l_ltki = new_l
Z_ltki = np.arange(0, Z_lt_max+1)

X, Y = np.meshgrid(l_ltki, Z_ltki)

plt.title("Late-Time kSZ Integrand \n(To Redshift z=6)")
plt.xlabel("l")
plt.ylabel("z")
plt.contourf(X, Y, ltki_cutoff, 100, cmap='RdGy')
plt.colorbar();


# ### Obtaining Reionization kSZ

# In[137]:


def ion_kSZ_integrand(z, kSZ_Bat, z_rei, sigma_rei):
    #z_rei and sigma_rei are constants, kSZ_Bat is the kSZ from Battaglia
    return ((np.exp((-(z-z_rei)**2)/(2*sigma_rei**2))/((2*np.pi*sigma_rei**2)**(1/2)))*kSZ_Bat)


z_rei = 8.8
sigma_rei = 1.0
ion_kSZ_integrand_arr = []
kSZ_Bat = Dl_kSZ_battaglia_interpolated_numpy
Z_rei_max = 20
Z_rei_min = 6

for i in range(Z_rei_max):
    ion_kSZ_integrand_arr.append(ion_kSZ_integrand(i, kSZ_Bat, z_rei, sigma_rei))

print(ion_kSZ_integrand_arr)

ion_kSZ_integrand_arr_add = np.resize(ion_kSZ_integrand_arr,(Z_rei_min,new_l_diff))

print(ion_kSZ_integrand_arr_add)

ion_kSZ_integrand_arr_append = ion_kSZ_integrand_arr[Z_rei_min:]

print(ion_kSZ_integrand_arr_append)


# ### Obtaining Full kSZ integrand

# In[177]:


full_kSZ_integrand_add = ltki_cutoff + ion_kSZ_integrand_arr_add
 

full_kSZ_integrand_append = np.concatenate((ltki_cutoff, ion_kSZ_integrand_arr_append))
print (len(full_kSZ_integrand_append))


# ### Obtaining dK_bar/dz

# In[178]:


dC_kSZ_dz = full_kSZ_integrand_append
dK_bar_integrand = []
for i in range(len(dC_kSZ_dz)):
    dK_bar_integrand_intermediate = []
    for k in range (2,len(dC_kSZ_dz[0])):
    #note, if involving the ell = 0 index, we get all nan output in dK_bar_dz, when involving ell = 1 index, we get all z = inf output in d_K_bar_dz 
        dK_bar_integrand_intermediate.append(dC_kSZ_dz[i][k]*(Ws[k]**2)*new_l[k]/(2*np.pi))
    dK_bar_integrand.append(dK_bar_integrand_intermediate)
        
print(len(dK_bar_integrand_intermediate))
print(len(dK_bar_integrand))
print(len(dK_bar_integrand[1]))

dK_bar_dz = []

for i in (range(len(dK_bar_integrand))):
    dK_bar_dz.append(np.trapz(dK_bar_integrand[i]))

print(dK_bar_dz)


# In[ ]:




