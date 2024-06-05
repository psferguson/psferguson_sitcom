import numpy as np
import matplotlib.pyplot as plt
import importlib.resources
import fitsio
import astropy.table
import scipy.interpolate as interpolate
import scipy.integrate as integrate

import fgcm
import lsst.utils

template_file = importlib.resources.files("fgcm.data.templates").joinpath("stellar_templates_master.fits")

fits = fitsio.FITS(template_file)
fits.update_hdu_list()
ext_names = []
for hdu in fits.hdu_list:
    ext_name = hdu.get_extname()
    if ('TEMPLATE_' in ext_name):
        ext_names.append(ext_name)

n_templates = len(ext_names)

templates = {}
for i in range(n_templates):
    templates[i] = fits[ext_names[i]].read(lower=True)
fits.close()

filters = ["u", "g", "r", "i", "z", "y"]
throughputs = {}
for filter_ in filters:
    tput = astropy.table.Table.read(f"https://raw.githubusercontent.com/lsst/throughputs/main/baseline/total_{filter_}.dat", format="ascii")
    tput.rename_column("col1", "wavelength")
    tput.rename_column("col2", "throughput")
    throughputs[filter_] = tput
for i in range(n_templates):
    for j, filter_ in enumerate(filters):
        template_lambda = templates[i]['lambda']
        template_f_lambda = templates[i]['flux']
        template_f_nu = template_f_lambda * template_lambda * template_lambda
        
        int_func = interpolate.interp1d(template_lambda, template_f_nu)
        tput_lambda = throughputs[filter_]["wavelength"]*10.
        f_nu = np.zeros(tput_lambda.size)
        # Make sure we interpolate in range
        good, = np.where((tput_lambda >= template_lambda[0]) & (tput_lambda <= template_lambda[-1]))
        f_nu[good] = int_func(tput_lambda[good])
        # out of range, let it hit the limit
        lo, = np.where(tput_lambda < template_lambda[0])
        f_nu[lo] = int_func(tput_lambda[good[0]])
        hi, = np.where(tput_lambda > template_lambda[-1])
        f_nu[hi] = int_func(tput_lambda[good[-1]])

        num = integrate.simpson(f_nu*throughputs[filter_]["throughput"]/tput_lambda, tput_lambda)
        denom = integrate.simpson(throughputs[filter_]["throughput"]/tput_lambda, tput_lambda)
        
        lsst_mags[j, i] = -2.5*np.log10(num/denom)

des_passbands = astropy.table.Table.read("https://noirlab.edu/science/sites/default/files/media/archives/documents/scidoc1884.txt", format="fits")
print(des_passbands)

des_filters = ["g", "r", "i", "z", "Y"]
des_mags = np.zeros((len(des_filters), n_templates))
for i in range(n_templates):
    for j, filter_ in enumerate(des_filters):
        template_lambda = templates[i]['lambda']
        template_f_lambda = templates[i]['flux']
        template_f_nu = template_f_lambda * template_lambda * template_lambda
        
        int_func = interpolate.interp1d(template_lambda, template_f_nu)
        tput_lambda = des_passbands["LAMBDA"]
        f_nu = np.zeros(tput_lambda.size)
        # Make sure we interpolate in range
        good, = np.where((tput_lambda >= template_lambda[0]) & (tput_lambda <= template_lambda[-1]))
        f_nu[good] = int_func(tput_lambda[good])
        # out of range, let it hit the limit
        lo, = np.where(tput_lambda < template_lambda[0])
        f_nu[lo] = int_func(tput_lambda[good[0]])
        hi, = np.where(tput_lambda > template_lambda[-1])
        f_nu[hi] = int_func(tput_lambda[good[-1]])

        num = integrate.simpson(f_nu*des_passbands[filter_]/tput_lambda, tput_lambda)
        denom = integrate.simpson(des_passbands[filter_], tput_lambda)
        
        des_mags[j, i] = -2.5*np.log10(num/denom)
des_filters = [i.lower() for i in des_filters]

des_rmi = des_mags[1, :] - des_mags[2, :]
des_gmi = des_mags[0, :] - des_mags[2, :]
des_imz = des_mags[2, :] - des_mags[3, :]

lsst_gmi = lsst_mags[2, :] - lsst_mags[3, :]
lsst_gmi = lsst_mags[2, :] - lsst_mags[3, :]
lsst_rmi = lsst_mags[1, :] - lsst_mags[3, :]
lsst_imz = lsst_mags[3, :] - lsst_mags[4, :]



for band in "grizy":
    if band in ["g","r","i"]:
        band_1="g"
        band_2="i"
        color = lsst_gmi
    elif band in ["z","y"]:
        band_1="i"
        band_2="z"
        color = lsst_imz
    
    xvals = color
    yvals = des_mags[des_filters.index(band),:] - lsst_mags[filters.index(band),:]
    yvals -=np.median(yvals)
    fit = np.polyfit(xvals,yvals,3)
    plt.figure()
    plt.scatter(xvals, yvals, c='r')
    plt.plot(xvals, np.polyval(fit,xvals))
    plt.xlabel(f"({band_1}-{band_2})_LSST")
    plt.ylabel(f"(DES - LSST_synth) {band}-band")
    print(f"(DES - LSST_synth) {band}-band: c1 + c2 * ({band_1}-{band_2})_lsst + c3 * ({band_1}-{band_2})_lsst^2 + c4 ({band_1}-{band_2})_lsst^3")
    print(f"fit: {fit[-1]:0.4f}, {fit[-2]:0.4f}, {fit[-3]:0.4f}, {fit[-4]:0.4f}")
    print(f"{band_1}-{band_2} range = {xvals.min():0.2f}, {xvals.max():0.2f}")
