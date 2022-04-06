import numpy as np 
from scipy import stats 
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt
import matplotlib.pyplot as plt 

def gauss_ring_logp(x,y,mean_radius=1,std=0.02):
    """
    Defines a gaussian ring likelihood on cartesian coordinater,
    around some ``mean_radius`` and with some ``std``.
    """
    circ = np.sqrt(x**2 + y**2)
    gauss = stats.norm.logpdf(circ, loc=mean_radius, scale=std)
    return gauss


def get_infodict():
    info = {"likelihood": 
        {
            "ring": gauss_ring_logp 
        }
    }

    info["params"] = {
        "x": {"prior": {"min": 0, "max": 2}, "ref": 0.5, "proposal": 0.01},
        "y": {"prior": {"min": 0, "max": 2}, "ref": 0.5, "proposal": 0.01}
    }

    info["params"]["r"] = {"derived": get_r}
    info["params"]["theta"] = {"derived": get_theta,
                            "latex": r"\theta", "min": 0, "max": np.pi/2}
    info["sampler"] = {"mcmc": {"Rminus1_stop": 0.001, "max_tries": 1000}}
    return info

def get_r(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def get_theta(x, y):
    return np.arctan(y / x)


def main():
    info = get_infodict()
    updated_info, sampler = run(info)
    gdsamples = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples, ["x", "y"], filled=True)
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.plots_1d(gdsamples, ["r", "theta"], nx=2)
    plt.show()

if __name__ == '__main__':
    main()