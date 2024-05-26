number_of_spectra = 100

#USE_LEGACY_CODE = True

import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
from QuasarCode import Settings

from podpy_refocused import Pod, TauBinned, UserInput, from_SpecWizard, fit_continuum, recover_c4, bin_pixels_from_SpecWizard, bin_combined_pixels_from_SpecWizard, BinnedOpticalDepthResults, plotting

# Load the comparison data
from paper_data_Q1317_0507 import plot_obs, plot_synthetic

Settings.enable_debug()

# Load the data (and recover H I)
spectrum_objects = from_SpecWizard()

# Recover metal optical depth (C IV)
fit_continuum(*spectrum_objects)
recover_c4(*spectrum_objects, observed_log10_flat_level = -3.05)

# Combine and bin pixels
results = bin_combined_pixels_from_SpecWizard("h1", "c4", *spectrum_objects, n_bootstrap_resamples = 0, legacy = False)



spectrum_objects_low_s_to_n = from_SpecWizard("/users/aricrowe/outputs/paper-1/EAGLE-L100N1504-SWF90/Turner16_QSO_Q1317-0507_high-noise/LongSpectrum.hdf5")
fit_continuum(*spectrum_objects_low_s_to_n)
recover_c4(*spectrum_objects_low_s_to_n, observed_log10_flat_level = -3.05)
results_low_s_to_n = bin_combined_pixels_from_SpecWizard("h1", "c4", *spectrum_objects_low_s_to_n, x_limits = (-1, 2.5), n_bootstrap_resamples = 0, legacy = False)



spectrum_objects_extreme_low_s_to_n = from_SpecWizard("/users/aricrowe/outputs/paper-1/EAGLE-L100N1504-SWF90/Turner16_QSO_Q1317-0507_extreme-noise/LongSpectrum.hdf5")
fit_continuum(*spectrum_objects_extreme_low_s_to_n)
recover_c4(*spectrum_objects_extreme_low_s_to_n, observed_log10_flat_level = -3.05)
results_extreme_low_s_to_n = bin_combined_pixels_from_SpecWizard("h1", "c4", *spectrum_objects_extreme_low_s_to_n, x_limits = (-1, 2.5), n_bootstrap_resamples = 0, legacy = False)



figure, axis = plotting.pod_statistics(results = results, label = "S/N = 100",#"This run",
                                       x_min = -1,
                                       x_max = 2.5,
                                       y_min = -3.5,
                                       y_max = 0.0,
                                       title = "T+16 Fig. 2 comparison -- Q1317-0507, z=3.7",
                                       x_label = "$\\rm log_{10}$ $\\tau_{\\rm H I}$",
                                       y_label = "Median $\\rm log_{10}$ $\\tau_{\\rm C IV}$")
figure, axis = plotting.pod_statistics(results = results_low_s_to_n, label = "S/N = 10",
                                       colour = "orange", linestyle = "--", hide_tau_min_label = True, allow_auto_set_labels = False, figure = figure, axis = axis)
figure, axis = plotting.pod_statistics(results = results_extreme_low_s_to_n, label = "S/N = 1",
                                       colour = "purple", linestyle = ":", hide_tau_min_label = True, allow_auto_set_labels = False, figure = figure, axis = axis)
plot_obs(label = "T+16 -- Obs. Data", axis = axis)
plot_synthetic(label = "T+16 -- EAGLE", axis = axis)
axis.legend()
plt.show()

exit()

single_spectrum_legacy = BinnedOpticalDepthResults.from_TauBinned(TauBinned(spectrum_objects[0].h1, spectrum_objects[0].c4, bsrs = 0, use_legacy_code = True))
single_spectrum_modern = BinnedOpticalDepthResults.from_TauBinned(TauBinned(spectrum_objects[0].h1, spectrum_objects[0].c4, bsrs = 0, use_legacy_code = False))
#binning_objects = bin_pixels_from_SpecWizard("h1", "c4", *spectrum_objects)
combined_legacy__single_spectrum_only = bin_combined_pixels_from_SpecWizard("h1", "c4", spectrum_objects[0], n_bootstrap_resamples = 0, legacy = True)
combined_legacy = bin_combined_pixels_from_SpecWizard("h1", "c4", *spectrum_objects, n_bootstrap_resamples = 0, legacy = True)
combined_updated = bin_combined_pixels_from_SpecWizard("h1", "c4", *spectrum_objects, n_bootstrap_resamples = 0, legacy = False)

figure, axis = plotting.pod_statistics(results = single_spectrum_legacy, label = "Single spectrum (legacy functions)",
                                       y_max = 0.0,
                                       title = "T+16 Fig. 2 comparison -- Q1317-0507, z=3.7",
                                       x_label = "$\\rm log_{10}$ $\\tau_{\\rm H I}$",
                                       y_label = "Median $\\rm log_{10}$ $\\tau_{\\rm C IV}$")
figure, axis = plotting.pod_statistics(results = single_spectrum_modern, label = "Single spectrum (modern functions)",
                                       colour = "orange", linestyle = "--", hide_tau_min_label = True, allow_auto_set_labels = False, figure = figure, axis = axis)
figure, axis = plotting.pod_statistics(results = combined_legacy__single_spectrum_only, label = "Single spectrum (modern code, legacy functions)",
                                       colour = "purple", linestyle = ":", hide_tau_min_label = True, allow_auto_set_labels = False, figure = figure, axis = axis)
figure, axis = plotting.pod_statistics(results = combined_legacy, label = "Combined spectra (modern code, legacy functions)",
                                       colour = "yellow", hide_tau_min_label = True, allow_auto_set_labels = False, figure = figure, axis = axis)
figure, axis = plotting.pod_statistics(results = combined_updated, label = "Combined spectra (modern code, modern functions)",
                                       colour = "black", linestyle = "--", hide_tau_min_label = True, allow_auto_set_labels = False, figure = figure, axis = axis)
plot_obs(label = "T+16 -- Obs. Data", axis = axis)
plot_synthetic(label = "T+16 -- EAGLE", axis = axis)
axis.legend()
#plt.show()
plt.savefig("update_test_plot.png")

exit()

tau_x = combined_bins["tau_binned_x"]
tau_y = combined_bins["tau_binned_y"]
tau_y_err = combined_bins["tau_binned_err"]
log_tau_min = combined_bins["tau_min"]

if log_tau_min is not None:
    plt.plot((tau_x[0], tau_x[-1]), (log_tau_min, log_tau_min), color = "blue", linestyle = ":", label = "$\\tau_{\\rm min}$")
#plt.plot(bin_centres[~np.isnan(median_logged_c4_opticaldepth)], median_logged_c4_opticaldepth[~np.isnan(median_logged_c4_opticaldepth)], color = "blue", label = "Test")
plt.plot(tau_x, tau_y, color = "blue", label = "This run")
plot_obs(label = "T+16 -- obs data")
plot_synthetic(label = "T+16 -- EAGLE ")
plt.title("T+16 Fig. 2 comparison -- Q1317-0507, z=3.7")
plt.xlabel("$\\rm log_{10}$ $\\tau_{\\rm H I}$")
plt.ylabel("Median $\\rm log_{10}$ $\\tau_{\\rm C IV}$")
plt.legend()
plt.xlim((tau_x[0], tau_x[-1]))
#plt.ylim((y_min, 0))

plt.show()
#if save:
#    plt.savefig("plot.png")
#plt.show()
#
#Console.print_info("DONE")

exit()



data = h5.File("LongSpectrum.hdf5")

wavelengths = data["Wavelength_Ang"][:]
fluxes = np.array([data[f"Spectrum{n}/Flux"][:] for n in range(1, 1 + number_of_spectra)])
flux_error_sigmas = np.array([data[f"Spectrum{n}/Noise_Sigma"][:] for n in range(1, 1 + number_of_spectra)])

spec_objects = []
pixel_wavelength_arrays = []
x_pixel_arrays = []
y_pixel_arrays = []
for i in range(number_of_spectra):
    spectrum = UserInput(3.7, "TestSpectrum", wavelengths, fluxes[i, :], flux_error_sigmas[i, :], verbose = False)
    spectrum.get_tau_rec_h1()
    spectrum.fit_continuum()
    spectrum.get_tau_rec_c4()

    binned_pixels = TauBinned(spectrum.h1, spectrum.c4) if i == 0 else None

    spec_objects.append((spectrum, binned_pixels))

    px_lambda, tau_x, tay_y = TauBinned._find_pixel_pairs(spectrum.h1, spectrum.c4)
    pixel_wavelength_arrays.append(px_lambda)
    x_pixel_arrays.append(tau_x)
    y_pixel_arrays.append(tay_y)

pixel_chunk_sizes = np.array([pixels.shape[0] for pixels in pixel_wavelength_arrays], dtype = int)
chunk_offsets = np.array([0, *np.cumsum(pixel_chunk_sizes)], dtype = int)
wavelength_data = np.empty(shape = pixel_chunk_sizes.sum(), dtype = np.float64)
x_data = np.empty_like(wavelength_data)
y_data = np.empty_like(wavelength_data)
for i in range(number_of_spectra):
    wavelength_data[chunk_offsets[i] : chunk_offsets[i + 1]] = pixel_wavelength_arrays[i]
    x_data[chunk_offsets[i] : chunk_offsets[i + 1]] = x_pixel_arrays[i]
    y_data[chunk_offsets[i] : chunk_offsets[i + 1]] = y_pixel_arrays[i]

spec_objects[0][1]._calc_percentiles(wavelength_data,
                                     x_data,
                                     y_data,
                                     tau_x_min = -5.75,
                                     tau_x_max = 4.0,
                                     bin_size = 0.25,)

#plt.scatter(x_data, y_data, s = 0.01)
plt.plot(spec_objects[0][1].tau_binned_x, spec_objects[0][1].tau_binned_y, c = "orange")
plt.show()
