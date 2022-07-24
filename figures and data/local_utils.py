import numpy as np
from scipy.optimize import curve_fit

def calc_zstar_from_slope_and_intercept(z0, slope, intercept):
    return z0*np.exp(-intercept/slope)

def calc_ustar_from_slope(slope, kappa=0.4):
    return kappa*slope

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

def calc_tilt(pitch, roll):
    # https://math.stackexchange.com/questions/2563622/vertical-inclination-from-pitch-and-roll
    return np.degrees(np.arctan(np.sqrt(np.tan(np.radians(roll))**2 +\
                                        np.tan(np.radians(pitch))**2)))

def chisqg(ydata,ymod,sd=None):
    """
    Returns the chi-square error statistic as the sum of squared errors between
    Ydata(i) and Ymodel(i). If individual standard deviations (array sd) are supplied,
    then the chi-square error statistic is computed as the sum of squared errors
    divided by the standard deviations.     Inspired on the IDL procedure linfit.pro.
    See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

    x,y,sd assumed to be Numpy arrays. a,b scalars.
    Returns the float chisq with the chi-square statistic.

    Rodrigo Nemmen
    http://goo.gl/8S1Oo
    """
    # Chi-square statistic (Bevington, eq. 6.9)
    if np.all(sd==None):
        chisq=np.sum((ydata-ymod)**2)
    else:
        chisq=np.sum( ((ydata-ymod)/sd)**2 )

    return chisq

def calc_analytic_sigma_intercept(sigma, N):
    return np.sqrt(2.*sigma**2*(2*N - 1)/N/(N + 1))

def calc_analytic_sigma_slope(delta_x, sigma, N):
    return np.sqrt(12*sigma**2/delta_x**2/N/(N**2 - 1))

def calc_sigma_intercept_slope(delta_x, sigma, N):
    return 6*sigma**2/N/(N + 1)/delta_x

def calc_analytic_sigma_ustar(delta_x, sigma, N, kappa=0.4):
    return kappa*calc_analytic_sigma_slope(delta_x, sigma, N)

def calc_analytic_sigma_zstar(z0, slope, intercept, delta_x, sigma, N, kappa=0.4):
    zstar = calc_zstar_from_slope_and_intercept(z0, slope, intercept)
    
    sigma_intercept = calc_analytic_sigma_intercept(sigma, N)
    sigma_slope = calc_analytic_sigma_slope(delta_x, sigma, N)
    sigma_intercept_slope = calc_sigma_intercept_slope(delta_x, sigma, N)
    
    return zstar*intercept/slope*np.sqrt((sigma_slope/slope)**2 +\
            (sigma_intercept/intercept)**2 -\
            2*(sigma_intercept_slope/intercept/slope))

def rescale_sigma(data, mod, sigma):
    #_NR_, 3rd ed, p. 783 - This equation provides a way to rescale
    # uncertainties, enforcing reduced chi-squared = 1

    redchisq = chisqg(data, mod, sd=sigma)

    return sigma*np.sqrt(redchisq/(len(data) - 2))

