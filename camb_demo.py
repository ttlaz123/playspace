import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats 
import sympy as sp
import time
import pickle 


from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt

#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower




print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))



global_var={}
def setup_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,As=2e-9, ns=0.965, r=0, lmax=2500, lens_acc=0):
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams( WantTensors=True)
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r= r)
    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_acc)
    return pars 

def plot_spectra(powers, compare='unlensed_scalar'):
    #plot the total lensed CMB power spectra versus unlensed, and fractional difference
    totCL=powers['total']
    unlensedCL=powers[compare]
    print(totCL.shape)
    print(unlensedCL.shape)
    #Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
    #The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
    ls = np.arange(totCL.shape[0])
    fig, ax = plt.subplots(2,2, figsize = (12,12))
    ax[0,0].plot(ls,totCL[:,0], color='k', label='Total')
    ax[0,0].plot(ls,unlensedCL[:,0], color='r', label=compare)
    ax[0,0].set_title('TT')
    ax[0,0].get_legend()
    ax[0,1].plot(ls[2:], 1-unlensedCL[2:,0]/totCL[2:,0]);
    ax[0,1].set_title(r'$\Delta TT$')
    ax[1,0].plot(ls,totCL[:,1], color='k')
    ax[1,0].plot(ls,unlensedCL[:,1], color='r')
    ax[1,0].set_title(r'$EE$')
    ax[1,1].plot(ls,totCL[:,3], color='k')
    try:
        ax[1,1].plot(ls,unlensedCL[:,3], color='r')
    except IndexError:
        pass
    ax[1,1].set_title(r'$TE$');
    for ax in ax.reshape(-1): 
        ax.set_xlim([2,2500])
    fig.suptitle('Comparing total with ' + str(compare))
    plt.show()

def simple_camb_demo():
    print('**********************')
    pars = setup_params(r=0.4)
    print('**********************')
    results = camb.get_results(pars)
    #get dictionary of CAMB power spectra
    print(results)
    print('**********************')
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    print(powers)
    print('**********************')
    for name in powers: 
        print(name)
        plot_spectra(powers, name)

def plot_cmb_transfer_l(trans, ix):
    _, axs = plt.subplots(1,2, figsize=(12,6))
    for source_ix, (name, ax) in enumerate(zip(['T', 'E'], axs)):
        ax.semilogx(trans.q,trans.delta_p_l_k[source_ix,ix,:])
        #ax.set_xlim([1e-5, 0.05])
        ax.set_xlabel(r'$k \rm{Mpc}$')
        ax.set_title(r'%s transfer function for $\ell = %s$'%(name, trans.L[ix]))
    plt.show()

def get_transfer_and_power(pars, lmax):
    print('getting results')
    results = camb.get_results(pars)
    print('getting powers')
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    print('getting transfer function data')
    pars.set_for_lmax(lmax=lmax-150) #idk why by the lmax is offset by 150
    data = camb.get_transfer_functions(pars)
    print('getting transfer mat')
    transfer = data.get_cmb_transfer_data()
    return powers, transfer

def power_k(ns, As, kstar, klist):
    '''
    Calculates P(k) = As*(k/kstar)^ns
    
    As, float
    ns, float
    klist, list of float
    kstar, float
    '''
    pk = np.zeros(klist.shape[0])
    pk = As*(klist/kstar)**(ns-1)
    return pk 

def gaussian_likelihood(x1, x2, sigma):
    '''
    Calculates gaussian likelihood given two points and a spread
    '''
    coeff = -np.log(2*np.pi*(sigma**2))/2
    exp = -(x1-x2)**2/(2*sigma**2)
    return coeff + exp

def likelihood_vector(measured, error, theory):
    '''
    Calculates Guassian likelihood of two list of points given a list of spreads
    Performs the vector operations to save computational time
    '''
    assert(len(measured) == len(error))
    assert(len(measured) == len(theory))
    loglikelihood=0

    coeff_list = -np.log(2*np.pi*(np.power(error,2)))/2
    coeff_sum = np.sum(coeff_list)

    diff_list = np.power(measured - theory, 2)
    sigma_list = 2*np.power(error, 2)
    exp_list = -np.divide(diff_list, sigma_list)
    exp_sum = np.sum(exp_list)

    return coeff_sum + exp_sum 


def likelihood_cl(measured, error, theory):
    '''
        calculates the likelihood of the theory given measured data and error bars
        assumes gaussian error bars
    '''
    assert(len(measured) == len(error))
    assert(len(measured) == len(theory))
    loglikelihood=0
    for i in range(len(measured)):
        loglikelihood += gaussian_likelihood(measured[i], theory[i], error[i])
    return loglikelihood 

def transfer_p_to_c(ks, pk, transfer):
    '''
        converts P(k) into C(l) using CAMB transfer functions
        TODO: currently takes about 0.02 sec
    '''
    assert(ks.shape[0] == pk.shape[0])
    assert(ks.shape[0] == transfer.shape[1])
    trans_squared = np.square(transfer)
    norm_pk = [pk[i]/ks[i] for i in range(len(ks))]

    integral = trans_squared.dot(norm_pk)
    cl = np.array([integral[i] * (i) * (i+1) for i in range(len(integral))])
    return cl 

def construct_line(x1, y1, x2, y2, xs):
    '''
    Given two points, produce a list of all ys given xs on a straight line

    x1, y1, x2, y2: floats
    xs: array of floats
    '''
    slope = (y2-y1)/(x2-x1)
    ys = slope*(xs-x1)+y1
    return ys


def construct_logspline(logk_list, logP_list, ks):
    '''
    Given coordinates in logspace, points are linearly connected in order

    logk_list: a list of k coordinates, log10, must be in ascending order
    logP_list: a list of P coordinates, natural log
    ks: a list of ks to fill in between the point list, linear space, 
            assumed ascending order, must cover all

    TODO: single run takes 0.068 sec, should try to speed up to 0.02 sec
    '''

    assert len(logk_list) == len(logP_list), "Number of Ks not equal to number of Ps"
    logks = np.log10(ks)
    assert logks[0] >= logk_list[0] and logks[-1] <= logk_list[-1],\
            "ks range not covered by klist"

    logPs = np.zeros(len(logks))
    index = 0
    for i in range(len(logk_list)-1):
        assert logk_list[i+1] > logk_list[i], "Ks not in ascending order"
    k = sp.symbols('k')
    piecewise = sp.interpolating_spline(1, k, logk_list, logP_list)
    spline_func = sp.lambdify(k,piecewise)
    logPs = spline_func(logks)
    return logPs



def spline_likelihood(logk_list, logP_list, err=None, measured=None, lmax=2400):
    '''
    Calculates the likelihood of an ns, As spline
    '''
    
    time1 = time.time()
    transfer = global_var['transfer']
    transfer_mat = transfer.delta_p_l_k[0]
    Ls = transfer.L 
    ks = transfer.q

    pk_spline = construct_logspline(logk_list, logP_list, ks)
    return pk_spline


def cosmos_likelihood(ns, As, err=None, measured=None,  lmax=2400):
    '''
        Calculates the likelihood of particular cosmological parameters given data
    '''
    
    pars = camb.CAMBparams()
    pars.set_for_lmax(lmax=lmax)
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=ns, As=As)
    pars.set_accuracy()
    #data = camb.get_transfer_functions(pars)
    #transfer = data.get_cmb_transfer_data()
    transfer = global_var['transfer']
    ks = transfer.q
    power_k = pars.scalar_power(ks)
    log_likelihood = power_spec_likelihood(power_spectrum, measured=measured, err=err)
    return log_likelihood

def get_infodict(measured):
    info = {"likelihood": 
        {
            "power":cosmos_likelihood 
        }
    }

    info["params"] = {
        "ns": {"prior": {"min": 0, "max": 1}, "ref": 0.5, "proposal": 0.01},
        "As": {"prior": {"min": 0, "max": 1e100}, "ref": 0.5, "proposal": 0.01},
        #"err": {"prior": {"min": 0, "max": 1e100}, "ref": 0.5, "proposal": 0.01}
    }

    info["sampler"] = {"mcmc": {"Rminus1_stop": 0.001, "max_tries": 1000}}
    return info


def get_default_instance(lmax, accuracy, lsample):
    '''
    Gives default transfer functions and 
    '''
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.set_accuracy(AccuracyBoost=accuracy, lSampleBoost=lsample)
    pars.set_for_lmax(lmax=lmax-40) # idk why it's offset by 50
    powers, transfer = get_transfer_and_power(pars, lmax)
    
    total_powers_T = powers['total'][:,0]
    print('Used multipoles: ' + str(transfer.L))
    print('Dimensions of Cl: ' + str(total_powers_T.shape))

    return transfer, total_powers_T 

def cobaya_demo_powerspectrum():
    '''
    will find ns and As
    '''
    transfer, total_powers = get_default_instance(lmax=2400)
    global_var['measured_power'] = total_powers
    global_var['transfer'] = transfer 

    print('Testing default likelihood: ' + str(cosmos_likelihood(0.96, 10)))

    info = get_infodict(total_powers)
    updated_info, sampler = run(info)
    gdsamples = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples, ["ns", "As"], filled=True)
    plt.show()

def power_spec_likelihood(power_spectrum, measured=None, err=None):
    '''
    Given a power spectrum in P(k), a power spectrum in Cl, and error bars, 
    calculates the likelihood of the theory using Gaussian spread
    
    power_spectrum: array of floats representing p(k), ks is in global_var
    measured: array of floats representing Cl, if is None then will obtain from global_var
    err: array of floats, if None will default to 100 for all L
    '''
    transfer = global_var['transfer']
    ks = transfer.q
    transfer_mat = transfer.delta_p_l_k[0]
    cl = transfer_p_to_c(ks, power_spectrum, transfer_mat)
    
    Ls = transfer.L 
    if(measured is None):
        measured=global_var['measured_power']
    measured = [measured[l] for l in Ls]
    if(err is None):
        err = [100 for i in range(len(Ls))]

    log_likelihood = likelihood_vector(measured, err, cl)
    return log_likelihood

def mcmc_spline_runner(k0, p0, 
                        k1=None, p1=None, 
                        k2=None, p2=None, 
                        k3=None, p3=None, 
                        k4=None, p4=None, 
                        k5=None, p5=None, 
                        k6=None, p6=None, 
                        k7=None, p7=None, 
                        k8=None, p8=None):
    '''
    Input:
     k0, k1, k2, ..., kn
     P0, P1, P2, ..., Pn
    
    finds likelihood based on spline points
    TODO: figure out how to pass in something like kwargs without cobaya freaking out
    '''

    numknots = 2
    logks_list = []
    logps_list = []
    logks_list.append(k0)
    logks_list.append(k1)
    logps_list.append(p0)
    logps_list.append(p1)

    if(k2 is not None):
        if(k2 < k1):
            raise AttributeError("Ks not in sorted order") 
        numknots += 1
        logks_list.append(k2)
        logps_list.append(p2)
    if(k3 is not None):
        if(k3 < k2):
            raise AttributeError("Ks not in sorted order") 
        numknots += 1
        logps_list.append(p3)
        logks_list.append(k3)
    if(k4 is not None):
        if(k4 < k3):
            raise AttributeError("Ks not in sorted order") 
        numknots += 1
        logps_list.append(p4)
        logks_list.append(k4)
    if(k5 is not None):
        if(k5 < k4):
            raise AttributeError("Ks not in sorted order") 
        numknots += 1
        logps_list.append(p5)
        logks_list.append(k5)
    if(k6 is not None):
        if(k6 < k5):
            raise AttributeError("Ks not in sorted order")  
        numknots += 1
        logps_list.append(p6)
        logks_list.append(k6)
    
    


    logspline = spline_likelihood(logks_list,logps_list)
    power_spectrum = np.exp(logspline)
    likelihood = power_spec_likelihood(power_spectrum)
    return likelihood

def get_spline_infodict():
    info = {"likelihood": 
        {
            "power":mcmc_spline_runner
        }
    }

    info["params"] = {
        "k0": -5.2,
        "k2": -0.3,
        "k1": {"prior": {"min": -5.2, "max": -0.3}, "ref": -3, "proposal": 0.01},
        "p0": {"prior": {"min": 0, "max": 3}, "ref": 2.99, "proposal": 0.01},
        "p1": {"prior": {"min": 0, "max": 3}, "ref": 2.99, "proposal": 0.01},
        "p2": {"prior": {"min": 0, "max": 3}, "ref": 0.5, "proposal": 0.01},
    }

    info["sampler"] = {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 1000}}
    info["output"] = "temp_output2"
    return info

def plot_info(info, sampler, outfile=None):
    gdsamples = MCSamplesFromCobaya(info, sampler.products()["sample"])
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples, ["k1", "p0", "p1", "p2"], filled=True)
    if(outfile is None):
        plt.show()
    else:
        plt.savefig(outfile)

def save_info(info, sampler, filename):
    with open(filename, 'wb') as f:
        pickle.dump((info, sampler), f)

def spline_driver():
    transfer, total_powers = get_default_instance(lmax=2000, accuracy=1, lsample=50)
    global_var['measured_power'] = total_powers
    global_var['transfer'] = transfer 
    # test to make sure the likelihood function works
    print(mcmc_spline_runner(k0=-5.2, k1=-3, k2=-0.3, p0=0.8, p1=0.9, p2=1.2))
    info_dict = get_spline_infodict()
    updated_info, sampler = run(info_dict, resume=True)
    return updated_info, sampler

def main():
    updated_info, sampler = spline_driver()
    filename='temp.pickle'
    plot_info(updated_info, sampler)
    
    

if __name__ == '__main__':
    main()