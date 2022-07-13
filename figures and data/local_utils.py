import numpy as np
from scipy.optimize import curve_fit

def calc_zstar_from_slope_and_intercept(z0, slope, intercept):
    return z0*np.exp(-intercept/slope)

def calc_ustar_from_slope(kappa, slope):
    return kappa*slope

def calc_sigma_ustar(kappa, sigma_slope):
    return kappa*sigma_slope

def calc_sigma_zstar(slope, zstar, sigma_slope, intercept, sigma_intercept, 
        slope_intercept_corr):
    return zstar/slope*np.sqrt(sigma_intercept**2 +\
            (intercept/slope)**2*sigma_slope**2 -\
            2*(intercept/slope)*slope_intercept_corr)

def calculate_zstar_from_profile(heights, winds):
    x = np.log(heights/np.min(heights))
    y = winds

    popt, pcov = curve_fit(lin_fit, x, y)

    zstar = calc_zstar_from_slope_and_intercept(np.min(heights), *popt)
    return zstar

def lin_fit(x, m, b):
    return m*x + b

def wind_profile(z, ustar, zstar, kappa=0.4):
    return ustar/kappa*np.log(z/zstar)

def fit_wind_profile(z, ustar, zstar, kappa=0.4):
    x = np.log(z/np.min(z))
    
    # slope
    m = ustar/kappa
    # intercept
    b = -ustar/kappa*np.log(zstar/np.min(z))
    
    return lin_fit(x, m, b)
