from astropy.io import fits
import numpy as np

def read_planck(filepath, spectrum_type):
    '''
    Returns the power spectrum as a function of l

    spectrum type 0 = TT, 1 = EE, 2 = BB, 3 = TE

    0: some weird image thing
    1: TT low l
    2: TE low l
    3: EE low l
    4: TB low l
    5: EB low l
    6: BB low l
    7: TT high l, binned
    8: TT high l, unbinned
    9: TE high l, binned
    10: TE high l, unbinned
    11: EE high l, binned
    12: EE high l, unbinned
    '''
    spectrum_data = fits.open(filepath)

    if(spectrum_type == 0):
        lowind = 1
        highind = 8
    elif(spectrum_type == 1):
        lowind = 3
        highind = 12
    elif(spectrum_type == 2):
        raise AttributeError('No data available for spectrum type ' + str(spectrum_type))
    elif(spectrum_type == 3):
        lowind = 2
        highind = 10
    else:
        raise AttributeError('No data available for spectrum type ' + str(spectrum_type))


    lowls = spectrum_data[lowind].data['ELL']
    lowdls = spectrum_data[lowind].data['D_ELL']
    #TODO have errs be more than just the average
    lowerrs = (spectrum_data[lowind].data['ERRUP'] + spectrum_data[lowind].data['ERRDOWN']) / 2

    hils = spectrum_data[highind].data['ELL']
    hidls = spectrum_data[highind].data['D_ELL']
    hierrs = spectrum_data[highind].data['ERR'] 

    ls = np.concatenate((lowls, hils))
    dls = np.concatenate((lowdls, hidls))
    errs = np.concatenate((lowerrs, hierrs))
    return ls, dls, errs
