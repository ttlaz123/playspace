import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats 
import sympy as sp
import time
import pickle

#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower, correlations
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

def power_k(ns, As, klist, kstar=1/20):
    '''
    Calculates P(k) = As*(k/kstar)^ns
    
    As, float
    ns, float
    klist, list of float
    kstar, float
    '''
    pk = As*(klist/kstar)**(ns-1)
    return pk 

def transfer_p_to_c(ks, pk, transfer, spectrum_type=0):
    '''
        converts P(k) into C(l) using CAMB transfer functions
        TODO: currently takes about 0.02 sec
    '''
    assert(ks.shape[0] == pk.shape[0])
    assert(ks.shape[0] == transfer.delta_p_l_k[0].shape[1])
    #alphas = alpha(l, 0, ks)
    delta_lnks = np.log(ks[1]/ks[0])
    norm_pk = pk/ks#*delta_lnks
    if(spectrum_type < 3):
        trans_squared = np.square(transfer.delta_p_l_k[spectrum_type])
   
        integral = trans_squared.dot(norm_pk)
        
        p = 0
        if(spectrum_type== 1 or spectrum_type ==2):
            p=3
    if(spectrum_type== 3):
        trans_squared = np.multiply(transfer.delta_p_l_k[0], transfer.delta_p_l_k[1])
        
        integral = trans_squared.dot(norm_pk)
        p=2
    ls = transfer.l
    cl = np.array([integral[i] * (ls[i]*(ls[i]+1))
                    #((transfer.l[i]+1)*(transfer.l[i]+2)/
                   # (transfer.l[i])/(transfer.l[i]-1) )**p
                   for i in range(len(integral))])
    #cl = integral
    return cl 

def beta(l, m, k, K=1):
    '''
    Returns the beta variable according to camb notes
    no idea what K is supposed to be
    '''
    beta = 1 - (l**2-(m+1)) * K/k**2
    return beta 

def alpha(l, m, ks):
    '''
    Returns the alpha variable according to camb notes
    Gives list of alphas as a function of k
    There might be a typo switching the locations of l and m?
    '''
    
    ns = np.array(range(m+1, l+1))
    betas = [np.prod(beta(ns, m, k)) for k in ks]
    return np.array(betas)


def get_transfer_functions(pars):
    '''
    
    '''
    data = camb.get_transfer_functions(pars)
    transfer = data.get_cmb_transfer_data()
    return transfer

def compare_transfer_funcs():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.96, As=2e-9)
    pars.set_accuracy(AccuracyBoost=1, lSampleBoost=50)
    pars.set_for_lmax(lmax=2000)
    transfer = get_transfer_functions(pars)
    print('Plotting...')
    print(transfer.q)
    print(transfer.delta_p_l_k.shape)
    a = transfer.delta_p_l_k[0][:, 500]
    
   

    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.96, As=2e-9)
    pars.set_accuracy(AccuracyBoost=10, lSampleBoost=50)
    pars.set_for_lmax(lmax=2000)
    transfer = get_transfer_functions(pars)
    print('Plotting...')
    print(transfer.q)
    print(transfer.delta_p_l_k.shape)
    b = transfer.delta_p_l_k[0][:, 500]
    plt.plot(b)
    plt.plot(a)
    plt.show()

def compare_transfer_results():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    time1 = time.time()
    results = camb.get_results(pars)
    time2= time.time()
    
    pars.set_for_lmax(lmax=2000)
    
    
    
    
    pars.InitPower.set_params(ns=0.96, As=2e-9)
    time3=time.time()
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    time4 = time.time()
    a=powers['unlensed_scalar'][:, 0]

    print('Getting results: ' + str(time2-time1))
    print('Getting powers: ' + str(time4-time3))

    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    #pars.InitPower.set_params(ns=0.96, As=2e-9)
    #pars.set_accuracy(AccuracyBoost=1, lSampleBoost=50)
    lmax=2000
    pars.set_for_lmax(lmax=lmax)
    pars.set_accuracy(AccuracyBoost=1, lSampleBoost=50)
    transfer = get_transfer_functions(pars)
    ks = transfer.q
    print(ks)
    print(transfer)
    ns = 0.96
    As = 2e-9
    pk = power_k(ns, As, ks)
    #inflation_params = initialpower.InitialPowerLaw()
    cl = transfer_p_to_c(ks, pk, transfer, spectrum_type=0)
    print(cl.shape)
    #inflation_params.set_params(ns=0.96, As=2e-9)
    #results.power_spectra_from_transfer(inflation_params) #warning OK here, not changing scalars
    #cl = results.get_total_cls(lmax, CMB_unit='muK')[:,0]
    print((cl*2e9)[0:100])
    
    a = [a[i] for i in range(2,len(a))]
    print(a[:100])
    div = a[0:2000]/cl[0:2000]
    print(np.polyfit(range(2000), div, 10))
    #plt.plot(div)
    plt.plot(a)
    plt.plot(transfer.l, cl*2e9)
    plt.show()

def main():
    print('Executing main')
    compare_transfer_results()
    #compare_transfer_funcs()
    print('Completed Main')
    
if __name__ == '__main__':
    main()
    