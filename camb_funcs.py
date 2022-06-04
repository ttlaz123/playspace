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

from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt

from fgivenx import plot_contours, samples_from_getdist_chains, plot_lines, plot_dkl

class MCMC_runner:
    def __init__(self, pars):
        self.pars = pars
        self.results = camb.get_results(pars)
global_var = {}
def likelihood_vector(measured, error, theory):
    '''
    Calculates Guassian likelihood of two list of points given a list of spreads
    Performs the vector operations to save computational time
    '''
    if(isinstance(error, float) or isinstance(error, int)):
        error = [error for i in range(len(measured))]

    assert(len(measured) == len(error))
    assert len(measured) == len(theory), "measured: " + str(len(measured)) + " theory: " + str(len(theory))
    loglikelihood=0
    '''
    while(measured[0]/measured[1] < 1e-3 or theory[0]/theory[1] < 1e-3):
        measured = measured[1:]
        theory = theory[1:]
        error = error[1:]
    '''
    coeff_list = -np.log(2*np.pi*(np.power(error,2)))/2
    coeff_sum = np.sum(coeff_list)

    diff_list = np.power(measured - theory, 2)
    sigma_list = 2*np.power(error, 2)
    exp_list = -np.divide(diff_list, sigma_list)
    exp_sum = np.sum(exp_list)
    return coeff_sum + exp_sum 

def power_k(klist, ns, As, kstar=1/20):
    '''
    Calculates P(k) = As*(k/kstar)^ns
    
    As, float
    ns, float
    klist, list of float
    kstar, float
    '''
    pk = As*(klist/kstar)**(ns-1)
    return pk

def construct_spline(ks, logk_list, logP_list):
    '''
    Exact same as construct_logspline, but converts to linear space
    '''
    logspline = construct_logspline(ks, logk_list, logP_list)
    return np.exp(logspline)

def construct_logspline(ks, logk_list, logP_list):
    '''
    Given coordinates in logspace, points are linearly connected in order

    logk_list: a list of k coordinates, log10, must be in ascending order
    logP_list: a list of P coordinates, natural log
    ks: a list of ks (or single k) to fill in between the point list, linear space, 
            assumed ascending order, must cover all

    TODO: single run takes 0.068 sec, should try to speed up to 0.02 sec
    '''
    assert len(logk_list) == len(logP_list), "Number of Ks not equal to number of Ps"
    logks = np.log10(ks)
    if(not isinstance(logks, float)):
        assert logks[0] >= logk_list[0] and logks[-1] <= logk_list[-1],\
                "ks range not covered by klist: " + str(logks)

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

def get_cls_from_pk(ns, As):
    args = (ns, As)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    lmax=2000
    pars.set_for_lmax(lmax,lens_potential_accuracy=1)
    pars.set_initial_power_function(power_k, args=args, 
                                    effective_ns_for_nonlinear=0.96)
    
    results = global_var['results']
    results.power_spectra_from_transfer(pars.InitPower)
    
    cl = results.get_lensed_scalar_cls( CMB_unit='muK')
    return cl

def get_cls_from_spline(logks_list, logps_list):
    args = (logks_list, logps_list)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    lmax=2000
    pars.set_for_lmax(lmax,lens_potential_accuracy=1)
    pars.set_initial_power_function(construct_spline, args=args, 
                                    effective_ns_for_nonlinear=0.96)
    
    results = global_var['results']
    results.power_spectra_from_transfer(pars.InitPower)
    
    cl = results.get_lensed_scalar_cls( CMB_unit='muK')
    return cl

def mcmc_spline_runner(spectrum_type, k0, p0, 
                        k1=None, p1=None, 
                        k2=None, p2=None, 
                        k3=None, p3=None, 
                        k4=None, p4=None, 
                        k5=None, p5=None, 
                        k6=None, p6=None, 
                        k7=None, p7=None, 
                        k8=None, p8=None, err=5., plot=True):
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

    cl = get_cls_from_spline(logks_list, logps_list)
    cl = cl[:, int(spectrum_type)]
    
    measured = global_var['measured']
    log_likelihood = likelihood_vector(measured, err, cl)
    
    if(plot):
        print(cl)
        plt.title('lensed_scalar')
        plt.xlabel('Multipole (l)')
        plt.ylabel('Power spectrum (Cl)')
        plt.plot(cl, label='Transfered')
        plt.plot(measured, label='Data')
        plt.legend()
        plt.show()
        print("Log likelihood: " + str(log_likelihood))
    return log_likelihood



def get_spline_infodict(output, spectrum_type):
    info = {"likelihood": 
        {
            "power":mcmc_spline_runner
        }
    }

    info["params"] = {
        "k0": -6,
        "k1": 2,
        #"k1": {"prior": {"min": -5.2, "max": -0.3}, "ref": -3, "proposal": 0.01},
        #"k2": {"prior": {"min": -5.2, "max": -0.3}, "ref": -2, "proposal": 0.01},
        "p0": {"prior": {"min": -30, "max": -19}, "ref": -25, "proposal": 0.01},
        "p1": {"prior": {"min": -30, "max": -19}, "ref": -25, "proposal": 0.01},
        #"p2": {"prior": {"min": 0, "max": 5}, "ref": 0.5, "proposal": 0.01},
        #"p3": {"prior": {"min": 0, "max": 5}, "ref": 0.5, "proposal": 0.01},
        'spectrum_type': spectrum_type,
        'err': 100,
        'plot': False
    }

    info["sampler"] = {"mcmc": {"Rminus1_stop": 0.1, "max_tries": 10000}}
    info["output"] = output
    return info

def spline_driver(output, spectrum_type):
    '''
    spectrum_type: 0=TT, 1=EE, 2=BB, 3=TE
    '''
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    results = camb.get_results(pars)
    global_var['results'] = results
    powers = results.get_lensed_scalar_cls(CMB_unit='muK')
    global_var['measured'] = powers[:, spectrum_type]
    # test to make sure the likelihood function works
    info_dict = get_spline_infodict(output, spectrum_type)
    init_params = info_dict['params']
    log_test = mcmc_spline_runner(spectrum_type=init_params['spectrum_type'], 
                                p0=init_params['p0']['ref'], 
                                p1=init_params['p1']['ref'],
                                #, p2=0.5, # k2=-0.3, ))
                                k0=init_params['k0'], 
                                k1=init_params['k1'], 
                                err=init_params['err'])
    print("test value: " + str(log_test))
    updated_info, sampler = run(info_dict, resume=True)
    return updated_info, sampler


def get_priors_and_variables(num_knots, kmin=-5.2, kmax=-0.3, pmin=-25, pmax=-15):
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


def plot_info(variables=None, info=None, sampler=None, outfile=None, file_root=None):
    '''
    info: yaml dictionary
    sampler: sampling output result from a cobaya run
    variables: list of variables to plot ['var1', 'var2', ...]
    outfile: path to location to save plot, if none, then simply shows plot
    '''
    if(not (info is None or sampler is None)):
        gdsamples = MCSamplesFromCobaya(info, sampler.products()["sample"])
    elif(not file_root is None):
        gdsamples = loadMCSamples(file_root)
    else:
        raise ValueError("No specified mc info")
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples, variables, filled=True)
    if(outfile is None):
        plt.show()
    else:
        plt.savefig(outfile)

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
def calc_logp(k, theta, k0=-6, kn=2):
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
    logp = construct_logspline( np.power(10., k), logk_list, logp_list)
    
    return logp 

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

    k = np.linspace(-6,2, 100)
    
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
def main():
    print('Executing main')
    output='camb_test/test1'
    spectrum_type=0
    num_knots = 2
    variables, priors = get_priors_and_variables(num_knots)
    updated_info, sampler = spline_driver(output, spectrum_type)
    plot_info(variables, updated_info, sampler)
    print(variables)
    print(priors)
    fgivenx_contours_logp(priors, variables, output)
    print('Completed Main')

def main2():
    spectrum_type=0
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    results = camb.get_results(pars)
    global_var['results'] = results
    powers = results.get_lensed_scalar_cls(CMB_unit='muK')
    global_var['measured'] = powers[:, spectrum_type]
    
    logks_list = [-6, 2]
    logps_list = [-19.57226, -20.35323]
    cl = get_cls_from_spline(logks_list, logps_list)
    cl = cl[:, int(spectrum_type)]

    cl2 = get_cls_from_pk(0.96, 2e-9)
    cl2 = cl2[:, int(spectrum_type)]
    plt.plot(cl)
    plt.plot(cl2)
    plt.show()

def main3():
    plot_info(variables=['As', 'ns'], file_root="chains/mcmc")

if __name__ == '__main__':
    main3()
    