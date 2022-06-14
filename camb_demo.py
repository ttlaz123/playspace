import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats 
import sympy as sp
import time
import pickle 

from cobaya.yaml import yaml_load
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt

#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower, correlations

from fgivenx import plot_contours, samples_from_getdist_chains, plot_lines, plot_dkl



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

def power_k(ns, As, klist, kstar=1/20):
    '''
    Calculates P(k) = As*(k/kstar)^ns
    
    As, float
    ns, float
    klist, list of float
    kstar, float (for some reason kstar is 1/20)
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
    assert len(measured) == len(theory), "measured: " + str(len(measured)) + " theory: " + str(len(theory))
    loglikelihood=0

    while(measured[0]/measured[1] < 1e-3 or theory[0]/theory[1] < 1e-3):
        measured = measured[1:]
        theory = theory[1:]
        error = error[1:]
    
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
    assert len(measured) == len(theory), "measured: " + str(len(measured)) + " theory: " + str(len(theory))
    loglikelihood=0
    for i in range(len(measured)):
        loglikelihood += gaussian_likelihood(measured[i], theory[i], error[i])
    return loglikelihood 

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
    if(not isinstance(logks, float)):
        assert logks[0] >= logk_list[0] and logks[-1] <= logk_list[-1],\
                "ks range not covered by klist"

    index = 0
    for i in range(len(logk_list)-1):
        assert logk_list[i+1] > logk_list[i], \
        ("Ks not in ascending order: " + 
        'k' + str(i+1) + ': ' + str(logk_list[i+1]) + ', k' + str(i) + ': ' + str(logk_list[i]))
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






def get_default_instance(lmax, accuracy, lsample, spectrum_type='TT'):
    '''
    Gives default transfer functions and 
    '''
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(r=0.01)
    pars.WantTensors=True
    pars.set_accuracy(AccuracyBoost=accuracy, lSampleBoost=lsample)
    pars.set_for_lmax(lmax=lmax-40) # idk why it's offset by 50
    powers, transfer = get_transfer_and_power(pars, lmax)
    

    ind = spectrum_type
    if(spectrum_type == 'TT'):
        ind = 0
    elif(spectrum_type == 'EE'):
        ind = 1
    elif(spectrum_type == 'BB'):
        ind = 2
    elif(spectrum_type == 'TE'):
        ind = 3
    elif(isinstance(spectrum_type, int)):
        ind = spectrum_type
    else:
        raise AttributeError('Spectrum type not correct: ' + str(spectrum_type))

    total_powers_T = powers['lensed_scalar'][:,ind]
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

def power_spec_likelihood(power_spectrum, measured=None, err=5., spectrum_type = 0, lensing=False, plot=False):
    '''
    Given a power spectrum in P(k), a power spectrum in Cl, and error bars, 
    calculates the likelihood of the theory using Gaussian spread
    
    power_spectrum: array of floats representing p(k), ks is in global_var
    measured: array of floats representing Cl, if is None then will obtain from global_var
    err: array of floats, if float, then will default to same err for all L
    '''
    
    transfer = global_var['transfer']
    ks = transfer.q
    spectrum_type = int(spectrum_type)
    transfer_mat = transfer.delta_p_l_k[spectrum_type]
    
    if(not isinstance(lensing, bool)):
        num_specs = 4
        clls = np.zeros(( len(transfer_mat[:,0]), num_specs))
        
        for i in range(num_specs):
            clls[:, i] = transfer_p_to_c(ks, power_spectrum, transfer, spectrum_type=i)
        cl = correlations.lensed_cls(clls,lensing[:,0])[:,spectrum_type]
    else:
        cl = transfer_p_to_c(ks, power_spectrum, transfer, spectrum_type=spectrum_type)
    
    Ls = transfer.L 
    if(measured is None):
        measured=global_var['measured_power']
    measured = [measured[l] for l in Ls]
    
    random_scale_factor = 2e9##TODO figure out why this is the case
    cl = cl*random_scale_factor

    if(isinstance(err, float) or isinstance(err, int)):
        err = [err for i in range(len(Ls))]
    log_likelihood = likelihood_vector(measured, err, cl)
    if(plot):
        print(cl)
        plt.title('lensed_scalar')
        plt.plot(cl[2:]-measured[2:])
        #plt.plot(cl, label='Transfered')
        #plt.plot(measured, label='Data')
        plt.legend()
        plt.show()
        print("Log likelihood: " + str(log_likelihood))
    return log_likelihood

def mcmc_spline_runner(spectrum_type, k0, p0, 
                        k1=None, p1=None, 
                        k2=None, p2=None, 
                        k3=None, p3=None, 
                        k4=None, p4=None, 
                        k5=None, p5=None, 
                        k6=None, p6=None, 
                        k7=None, p7=None, 
                        k8=None, p8=None, err=5.):
    '''
    Input:
     k0, k1, k2, ..., kn
     P0, P1, P2, ..., Pn
    
    finds likelihood based on spline points
    TODO: figure out how to pass in something like kwargs without cobaya freaking out
    currently takes about 0.075 secs to run
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
            return -1e100
        numknots += 1
        logks_list.append(k2)
        logps_list.append(p2)
    if(k3 is not None):
        if(k3 < k2):
            return -1e100
            raise AttributeError("Ks not in sorted order") 
        numknots += 1
        logps_list.append(p3)
        logks_list.append(k3)
    if(k4 is not None):
        if(k4 < k3):
            return -1e100
            raise AttributeError("Ks not in sorted order") 
        numknots += 1
        logps_list.append(p4)
        logks_list.append(k4)
    if(k5 is not None):
        if(k5 < k4):
            return -1e100
            raise AttributeError("Ks not in sorted order") 
        numknots += 1
        logps_list.append(p5)
        logks_list.append(k5)
    if(k6 is not None):
        if(k6 < k5):
            return -1e100
            raise AttributeError("Ks not in sorted order")  
        numknots += 1
        logps_list.append(p6)
        logks_list.append(k6)
    
    


    logspline = spline_likelihood(logks_list,logps_list)
    power_spectrum = np.exp(logspline)
    
    likelihood = power_spec_likelihood(power_spectrum, spectrum_type=int(spectrum_type), err=err, plot=False)
    return likelihood



def plot_info(info, sampler, variables, outfile=None):
    '''
    info: yaml dictionary
    sampler: sampling output result from a cobaya run
    variables: list of variables to plot ['var1', 'var2', ...]
    outfile: path to location to save plot, if none, then simply shows plot
    '''
    gdsamples = MCSamplesFromCobaya(info, sampler.products()["sample"])
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples, variables, filled=True)
    if(outfile is None):
        plt.show()
    else:
        plt.savefig(outfile)

def save_info(info, sampler, filename):
    with open(filename, 'wb') as f:
        pickle.dump((info, sampler), f)



def calc_logp(k, theta, k0=-5.2, kn=-0.3):
    '''
    Provides a value for the logspline depending on the parameters given in theta

    Assumes parameters in theta are in the order [p0, k1, p1, k2 ... kn-1, pn-1, pn]
    '''
    
    num_params = len(theta)
    assert(num_params %2 == 0)
    num_ps = int(num_params/2 + 1)
    num_ks = int(num_params/2 - 1)

    logp_list = []
    logk_list = [k0]
    
    for i in range(num_params):
        if(i == num_params -1 ):
            logp_list.append(theta[i])
        elif(i % 2 == 0):
            logp_list.append(theta[i])
        else:
            logk_list.append(theta[i])
    logk_list.append(kn)

    
    logp = construct_logspline(logk_list, logp_list, np.power(10., k))
    return logp 

def get_uniform_sample(range_list, variables, nsamples=100):
    '''
    range_list: list of (min, max)
    variables: [p0, k1, p1, ... kn-1, pn-1, pn]
    '''
    samples = np.zeros((nsamples, len(range_list)))
    for i, r in enumerate(range_list):
        samples[:,i] = np.random.uniform(low=r[0], high=r[1], size=nsamples)
    
    #TODO figure out how to sort the ks
    k_inds = []

    for i, v in enumerate(variables):
        if(v[0] == 'k'):
            k_inds.append(i)
    k_inds = np.array(k_inds)
    if(len(variables) == 2):
        return samples
    for i in range(samples.shape[0]):
        k_samples = samples[i, k_inds]
        samples[i, k_inds] = np.sort(k_samples)
    
    return samples 

def fgivenx_contours_logp(priors, variables, file_root):
    '''
    Plots the likelihood contours for log of the power spectrum with the fgivenx library

    priors: a list of (min, max) for priors on the variables
    variables: a list of strings indicating which variables ran through the mcmc in the oder
                corresponding to the priors 
    file_root: the folder/prefix for the mcmc output
    '''
    

    samples, weights = samples_from_getdist_chains(variables, file_root)
    
    uniform_samples = get_uniform_sample(priors, variables=variables)
    uniform_weights = np.ones(len(uniform_samples))

    k = np.linspace(-5.1, -0.4, 100)
    
    fig, axes = plt.subplots(1, 2)

    axes[0].set_xlim([min(k), max(k)])
    axes[0].set_ylim([priors[0][0],priors[0][-1]])
    axes[1].set_xlim([min(k), max(k)])
    axes[1].set_ylim([priors[0][0],priors[0][-1]])

    axes[0].set_xlabel('Log(k) Wavenumber')
    axes[1].set_xlabel('Log(k) Wavenumber')
    axes[0].set_ylabel('Log(P) Power')

    plot_contours(calc_logp, k, uniform_samples, weights=uniform_weights, 
                    ax=axes[1], colors=plt.cm.Blues_r, lines=False)
    cbar = plot_contours(calc_logp, k, samples, weights=weights, 
                    ax=axes[1], colors=plt.cm.Reds_r)
    
    plot_lines(calc_logp, k, uniform_samples, weights=uniform_weights, ax=axes[0], color='b')
    plot_lines(calc_logp, k, samples, weights=weights, ax=axes[0], color='r')

    cbar = plt.colorbar(cbar,ticks=[0,1,2,3])
    cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$'])
 
    plt.show()

def get_priors_and_variables(num_knots, kmin=-5.2, kmax=-0.3, pmin=-30, pmax=-20):
    '''
    returns variables = [p0, k1, p1, ... kn-1, pn-1, pn]
            priors = [(min, max), (min, max), etc]
    '''
    priors_k = [kmin, kmax]
    priors_p = [pmin, pmax]

    variables = []
    priors = []
    for i in range(num_knots):
        pvar = 'p' + str(i)
        kvar = 'k' + str(i)
        if(i == 0 or i == num_knots -1):
            variables.append(pvar)
            priors.append(priors_p)
        else:
            variables.append(kvar)
            priors.append(priors_k)
            variables.append(pvar)
            priors.append(priors_p)
            
    
    return variables, priors 

def lensing_test():
    #Plot CMB lensing potential power for various values of w
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    
    pars.set_for_lmax(2000, lens_potential_accuracy=1)
    
    w=-1
    pars.set_dark_energy(w=w, wa=0, dark_energy_model='fluid') 
    
    for As in [1e-9, 2e-9, 3e-9]:
        pars.InitPower.set_params(As=As, ns=0.965)
        time0 = time.time()
        
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK',lmax=2000)
        print(powers['unlensed_total'].shape)
        cl = results.get_lens_potential_cls(lmax=2000)
        print(cl[:,0].shape)
        lensed_cl = correlations.lensed_cls(powers['unlensed_total'],cl[:,0])
        print('time: ' + str(time.time()-time0))
        plt.plot(np.arange(2001), lensed_cl[:,1], label='lensed: ' + str(As))
        plt.plot(np.arange(2001), powers['unlensed_total'][:,1], label='unlensed: ' + str(As))

    #plt.legend([1e-9, 2e-9, 3e-9])
    plt.legend()
    plt.ylabel('$[L(L+1)]^2C_L^{\phi\phi}/2\pi$')
    plt.xlabel('$L$')
    plt.xlim([2,2000])
    plt.show()

def spline_driver(output, spectrum_type):
    '''
    spectrum_type: 0=TT, 1=EE, 2=BB, 3=TE
    '''
    transfer, total_powers = get_default_instance(lmax=2000, accuracy=1, lsample=50, spectrum_type=spectrum_type)
    global_var['measured_power'] = total_powers
    global_var['transfer'] = transfer 
    # test to make sure the likelihood function works
    print(mcmc_spline_runner(spectrum_type=spectrum_type, p0=-28, p1=-25,k0=-5.2, k1=-0.3, err=0.01))#, p2=0.5, # k2=-0.3, ))
    info_dict = yaml_load(output)#get_spline_infodict(output, spectrum_type)
    updated_info, sampler = run(info_dict, resume=True)
    return updated_info, sampler

def cosmos_likelihood(ns, As, err=5, measured=None, lensing=False, lmax=2000, plot=True):
    '''
        Calculates the likelihood of particular cosmological parameters given data
    '''
    
    pars = camb.CAMBparams()
    pars.set_for_lmax(lmax=lmax)
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    
    pars.set_accuracy()
    #data = camb.get_transfer_functions(pars)
    #transfer = data.get_cmb_transfer_data()
    transfer = global_var['transfer']
    ks = transfer.q
    pars.InitPower.set_params(ns=ns, As=As)
    power_k = pars.scalar_power(ks)
    if(lensing):
        
        results = camb.get_results(pars)
        lensing = results.get_lens_potential_cls(lmax=lmax)
    #print(lensing)
    log_likelihood = power_spec_likelihood(power_k, measured=measured, err=err, lensing=lensing, plot = plot)
    return log_likelihood

def get_infodict(output, spectrum_type):
    info = {"likelihood": 
        {
            "power":cosmos_likelihood 
        }
    }

    info["params"] = {
        "ns": {"prior": {"min": 0, "max": 1}, "ref": 0.5, "proposal": 0.01},
        "As": {"prior": {"min": 0, "max": 1e-8}, "ref": 2e-9, "proposal": 1e-11},
        #"err": {"prior": {"min": 0, "max": 1e100}, "ref": 0.5, "proposal": 0.01}
        #'spectrum_type': spectrum_type,
        'err': 100,
        'plot': False,
        #'lensing': False,
    }

    info["sampler"] = {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 10000}}
    info["output"] = output
    return info

def get_spline_infodict(output, spectrum_type):
    info = {"likelihood": 
        {
            "power":mcmc_spline_runner
        }
    }

    info["params"] = {
        "k0": -5.2,
        "k1": -0.3,
        "k1": {"prior": {"min": -5.2, "max": -0.3}, "ref": -3, "proposal": 0.01},
        #"k2": {"prior": {"min": -5.2, "max": -0.3}, "ref": -2, "proposal": 0.01},
        "p0": {"prior": {"min": 0, "max": 5}, "ref": 2.99, "proposal": 0.01},
        "p1": {"prior": {"min": 0, "max": 5}, "ref": 2.99, "proposal": 0.01},
        "p2": {"prior": {"min": 0, "max": 5}, "ref": 0.5, "proposal": 0.01},
        #"p3": {"prior": {"min": 0, "max": 5}, "ref": 0.5, "proposal": 0.01},
        'spectrum_type': spectrum_type,
        #'err': 0.01
    }

    info["sampler"] = {"mcmc": {"Rminus1_stop": 1, "max_tries": 1000}}
    info["output"] = output
    info["resume"] = True
    return info

def main():
    file_root = 'out3_tt/3knots_tt'
    spectrum_type = 0
    transfer, total_powers = get_default_instance(lmax=2000, accuracy=1, lsample=50, spectrum_type=0)
    global_var['measured_power'] = total_powers
    global_var['transfer'] = transfer 

    info_dict = get_spline_infodict(file_root, spectrum_type)
    variables, priors = get_priors_and_variables(2)
    #print(cosmos_likelihood(ns=7.611533, As=2.826542e-12, err=100))
    #print(cosmos_likelihood(ns=0.955, As=1.93e-9, err=100, lensing=True))
    #updated_info, sampler = run(info_dict, resume=True)
    #variables=('ns','As')
    updated_info, sampler = spline_driver(file_root, spectrum_type=spectrum_type)
    plot_info(updated_info, sampler, variables)
    
    
    print(variables)
    print(priors)
    fgivenx_contours_logp(priors, variables, file_root)

def main2():
    lmax = 2000
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    ns = 0.96
    As = 2e-9
    kp=1/20
    pars.InitPower.set_params(ns=ns, As=As)
    #pars.WantTensors=False
    pars.set_accuracy(lSampleBoost=50)
    pars.set_for_lmax(lmax=lmax-40) # idk why it's offset by 40
    results = camb.get_results(pars)
    print('getting powers')
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    print('getting transfer function data')
    pars.set_for_lmax(lmax=lmax-150) #idk why by the lmax is offset by 150
    data = camb.get_transfer_functions(pars)
    print('getting transfer mat')
    transfer = data.get_cmb_transfer_data()
    ks = transfer.q
    total_powers_T = powers['lensed_scalar'][:,0]
    pk = pars.scalar_power(ks)
    norm_pk = [pk[i]/ks[i] for i in range(len(ks))]
    spectrum_type=0
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
    
    cl = np.array([integral[i] * ((i) * (i+1))**p
                    for i in range(len(integral))])
    
    
    ind = spectrum_type
    total_powers_T = powers['lensed_scalar'][:,ind]
    print(powers.keys())
    #plt.plot((powers['unlensed_total'][:,ind]-powers['unlensed_scalar'][:,ind])[1:], label='total')
    

    plt.errorbar(range(len(powers['total'][:,ind])),powers['total'][:,ind], yerr = 100, label='Mock Data with Error Bars')
    plt.plot(transfer.L, cl*2e9, label = 'Computed from Transfer Function', linewidth=3)
    plt.xlabel('Multipole')
    plt.ylabel('l(l+1)C^TT')
    plt.title('Fitting Power Spectrum to Raw Data')
    #plt.plot(powers['unlensed_scalar'][:,ind], label='unlensed_scalar')
    #plt.plot(powers['unlensed_total'][:,ind], label='unlensed_total')
    #plt.plot(powers['lensed_scalar'][:,ind], label='lensed_scalar')
    #plt.plot(powers['tensor'][:,ind], label='tensor')
    #plt.plot(powers['lens_potential'][:,ind], label='lens_potential')
    
    plt.legend()
    plt.show()
    plt.semilogy(powers['unlensed_scalar'][0:1999,ind]/cl/2e9, label='ratio between unlensed')
    plt.semilogy(powers['lensed_scalar'][0:1999,ind]/cl/2e9, label = 'ratio between lensed')
    plt.plot([0, 1999],[1,1], label = 'y=1')
    plt.xlabel('Multipole (l)')
    plt.ylabel('Power Cl')
    plt.legend()
    plt.show()
    

'''

    plt.xlabel('Multiple l')
    plt.ylabel('Anisotropy Cl (uK^2)')
    plt.title("CMB Power Spectrum with different parameters")
    plt.legend()
    plt.show()
'''

'''
plt.figure(figsize=(6,6))
    plt.imshow(np.clip(trans_squared,1e-30, 1e-16),norm=matplotlib.colors.LogNorm())
    plt.xticks([0, 2000],[np.format_float_scientific(ks[0], 2),np.format_float_scientific(ks[2000], 2)])
    plt.xlabel('Wavenumber kMpc')
    plt.ylabel('Multipole l')
    plt.title('Transfer Function Squared')
    cbar = plt.colorbar()
    cbar.set_label('Transfer Function Value')
    plt.show()

'''
if __name__ == '__main__':
    main()