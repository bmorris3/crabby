import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from crabby import (generate_master_flat_and_dark, photometry,
                     PhotometryResults, PCA_light_curve, params_e,
                     transit_model_e)

# Image paths
image_paths = sorted(glob('/Users/bmmorris/data/saintex/2019_55cnc/20190430/Raw/*.fts'))[:-2]
dark_paths = glob('/Users/bmmorris/data/saintex/2019_55cnc/20190430/Dark/*.fts')
flat_paths = glob('/Users/bmmorris/data/saintex/2019_55cnc/20190430/Flat/*.fts')
master_flat_path = 'outputs/masterflat_20190430.fits'
master_dark_path = 'outputs/masterdark_20190430.fits'

# Photometry settings
target_centroid = np.array([[1111], [1111]])
comparison_flux_threshold = 0.01
aperture_radii = np.arange(20, 30, 2)
centroid_stamp_half_width = 30
psf_stddev_init = 30
aperture_annulus_radius = 10
transit_parameters = params_e
star_positions = np.array([[1092, 1092], [1320, 1815]])

output_path = 'outputs/55cnc_20190430.npz'
force_recompute_photometry = False #True

# Calculate master dark/flat:
if not os.path.exists(master_dark_path) or not os.path.exists(master_flat_path):
    print('Calculating master flat:')
    generate_master_flat_and_dark(flat_paths, dark_paths,
                                  master_flat_path, master_dark_path)

# Do photometry:

if not os.path.exists(output_path) or force_recompute_photometry:
    print('Calculating photometry:')
    phot_results = photometry(image_paths, master_dark_path, master_flat_path,
                              target_centroid, comparison_flux_threshold,
                              aperture_radii, centroid_stamp_half_width,
                              psf_stddev_init, aperture_annulus_radius,
                              output_path, star_positions)

else:
    phot_results = PhotometryResults.load(output_path)

print(phot_results.xcentroids.shape)

# print('Calculating PCA...')

# plt.plot(phot_results.times, phot_results.fluxes[:, 1, :5])
# plt.show()


lcs = phot_results.fluxes[:, 0, :]/phot_results.fluxes[:, 1, :]
std = np.std(lcs, axis=0)

start = 50
stop = -10
light_curve = lcs[start:stop, std.argmax()]

not_cloudy = np.ones_like(light_curve).astype(bool)


minus_airmass = 1 - phot_results.airmass[start:stop]
dx = phot_results.xcentroids[start:stop, 0] - phot_results.xcentroids[:, 0].mean()
dy = phot_results.ycentroids[start:stop, 0] - phot_results.ycentroids[:, 0].mean()
X = np.vstack([light_curve,
               minus_airmass,
               minus_airmass**2,
               dx, dy,
               phot_results.background_median[start:stop]
               ]).T

c = np.linalg.lstsq(X, np.ones(X.shape[0]))[0]

detrended_light_curve = X @ c

plt.plot(phot_results.times[start:stop], detrended_light_curve, '.', color='gray')

plt.show()