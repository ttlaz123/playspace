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

def transfer_p_to_c(ks, pk, transfer, spectrum_type=0):
    '''
        converts P(k) into C(l) using CAMB transfer functions
        TODO: currently takes about 0.02 sec
    '''
    assert(ks.shape[0] == pk.shape[0])
    assert(ks.shape[0] == transfer.delta_p_l_k[0].shape[1])
    norm_pk = [pk[i]/ks[i] for i in range(len(ks))]
    if(spectrum_type < 3):
        trans_squared = np.square(transfer.delta_p_l_k[spectrum_type])
   
        integral = trans_squared.dot(norm_pk)
        
        p = 1
        if(spectrum_type== 1 or spectrum_type ==2):
            p=3
    if(spectrum_type== 3):
        trans_squared = np.multiply(transfer.delta_p_l_k[0], transfer.delta_p_l_k[1])
        
        integral = trans_squared.dot(norm_pk)
        p=2

    cl = np.array([integral[i] * ((i) * (i+1))**p #*((l+1)*(l+2)/(l-1)/l)**(p/2)
                    for i in range(len(integral))])
    return cl 

def get_transfer_functions(pars):
    
    data = camb.get_transfer_functions(pars)
    transfer = data.get_cmb_transfer_data()
    return transfer

def main():
    print('Executing main')
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.96, As=2e-9)
    pars.set_accuracy(AccuracyBoost=1, lSampleBoost=50)
    pars.set_for_lmax(lmax=2000)
    transfer = get_transfer_functions(pars)
    print('Plotting...')
    print(transfer.q)
    print(transfer.delta_p_l_k.shape)
    a = transfer.delta_p_l_k[0][:, -1]
    
   

    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.96, As=2e-9)
    pars.set_accuracy(AccuracyBoost=2, lSampleBoost=50)
    pars.set_for_lmax(lmax=2000)
    transfer = get_transfer_functions(pars)
    print('Plotting...')
    print(transfer.q)
    print(transfer.delta_p_l_k.shape)
    b = transfer.delta_p_l_k[0][:, -1]
    plt.plot(b)
    plt.plot(a)
    plt.show()

if __name__ == '__main__':
    main()
    