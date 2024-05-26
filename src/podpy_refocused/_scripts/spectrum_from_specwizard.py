# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
import numpy as np
from QuasarCode import Settings, Console
from QuasarCode.Tools import ScriptWrapper
from typing import Union, List
from matplotlib import pyplot as plt
import h5py as h5

from .._Spectrum import from_SpecWizard, fit_continuum, recover_c4, recover_o6
from .._TauBinned import bin_combined_pixels_from_SpecWizard, BinnedOpticalDepthResults
from .. import plotting
from .._universe import c as speed_of_light_kms

def __main(
           datafile: str,
           raw: bool,
           h1: bool,
           c4: bool,
           o6: bool,
           target_spectrum: int,
           wavelength: bool,
           velocity: bool,
           redshift: bool,
           x_min: Union[float, None],
           x_max: Union[float, None],
           no_h1_corrections: bool,
           no_metal_corrections: bool,
           compare_corrections: bool,
           use_RSODOST_field: bool,
           title: Union[str, None],
           output_file: Union[str, None]
        ) -> None:
    
    if not (raw or h1 or c4 or o6):
        Console.print_error("Target ion(s) must be specified!")
        Console.print_info("Terminating...")
        return

    if not (wavelength or velocity or redshift):
        Console.print_error("Spectrum X-axis type must be specified! (options are '--wavelength', '--velocity' or '--redshift')")
        Console.print_info("Terminating...")
        return
    
    # Load and process data

    x_data = []
    flux_data = []
    labels = []
    colours = []
    linestyles = []
    alphas = []

    data = h5.File(datafile)
    first_spec_num = int(data["Parameters/SpecWizardRuntimeParameters"].attrs["first_specnum"])
    n_spectra = int(data["Parameters/SpecWizardRuntimeParameters"].attrs["NumberOfSpectra"])
    if (target_spectrum >= 0 and target_spectrum >= n_spectra) or (target_spectrum < 0 and -target_spectrum > n_spectra):
        raise IndexError(f"Unable to select sightline index {target_spectrum} as there are only {n_spectra} spectra avalible.")
    selected_spectrum_data = data[f"Spectrum{first_spec_num + target_spectrum}"]

    if raw:
        x_data.append(data["Wavelength_Ang"][:])
        flux_data.append(selected_spectrum_data["Flux"][:])
        labels.append("Raw Flux")
        colours.append("grey")
        linestyles.append("-")
        alphas.append(1.0)

    if not use_RSODOST_field:

        uncorrected_h1_wavelength = None
        uncorrected_c4_wavelength = None
        uncorrected_o6_wavelength = None
        uncorrected_h1_redshift = None
        uncorrected_c4_redshift = None
        uncorrected_o6_redshift = None

        uncorrected_h1_tau = None
        uncorrected_c4_tau = None
        uncorrected_o6_tau = None

        corrected_h1_wavelength = None
        corrected_c4_wavelength = None
        corrected_o6_wavelength = None
        corrected_h1_redshift = None
        corrected_c4_redshift = None
        corrected_o6_redshift = None

        corrected_h1_tau = None
        corrected_c4_tau = None
        corrected_o6_tau = None

        if h1 or c4:
            dla_masked_spectrum_object = from_SpecWizard(
                                                            filepath = datafile,
                                                            sightline_filter = [target_spectrum],
                                                            identify_h1_contamination = compare_corrections or not no_h1_corrections,
                                                            correct_h1_contamination = compare_corrections or not no_h1_corrections,
                                                            mask_dla = True
                                                        )[0]
            if c4:
                if compare_corrections or not no_metal_corrections:
                    recover_c4(dla_masked_spectrum_object, apply_recomended_corrections = True)
                    corrected_c4_wavelength = dla_masked_spectrum_object.c4.lambdaa
                    corrected_c4_redshift = dla_masked_spectrum_object.c4.z
                    corrected_c4_tau = dla_masked_spectrum_object.c4.tau_rec
                if compare_corrections or no_metal_corrections:
                    recover_c4(dla_masked_spectrum_object, apply_recomended_corrections = False)
                    uncorrected_c4_wavelength = dla_masked_spectrum_object.c4.lambdaa
                    uncorrected_c4_redshift = dla_masked_spectrum_object.c4.z
                    uncorrected_c4_tau = dla_masked_spectrum_object.c4.tau_rec
            
            if h1:
                if compare_corrections or not no_h1_corrections:
                    corrected_h1_wavelength = dla_masked_spectrum_object.h1.lambdaa
                    corrected_h1_redshift = dla_masked_spectrum_object.h1.z
                    corrected_h1_tau = dla_masked_spectrum_object.h1.tau_rec
                    if compare_corrections:
                        # Compute the uncorrected H I optical depths
                        uncorrected_h1_spectrum_object = from_SpecWizard(
                                                                        filepath = datafile,
                                                                        sightline_filter = [target_spectrum],
                                                                        identify_h1_contamination = False,
                                                                        correct_h1_contamination = False,
                                                                        mask_dla = True
                                                                    )[0]
                        uncorrected_h1_wavelength = uncorrected_h1_spectrum_object.h1.lambdaa
                        uncorrected_h1_redshift = uncorrected_h1_spectrum_object.h1.z
                        uncorrected_h1_tau = uncorrected_h1_spectrum_object.h1.tau_rec
                else:
                    uncorrected_h1_wavelength = dla_masked_spectrum_object.h1.lambdaa
                    uncorrected_h1_redshift = dla_masked_spectrum_object.h1.z
                    uncorrected_h1_tau = dla_masked_spectrum_object.h1.tau_rec

        if o6:
            non_dla_masked_spectrum_object = from_SpecWizard(
                                filepath = datafile,
                                sightline_filter = [target_spectrum],
                                identify_h1_contamination = not no_h1_corrections,
                                correct_h1_contamination = not no_h1_corrections,
                                mask_dla = False
                            )[0]
            if compare_corrections or not no_metal_corrections:
                recover_o6(non_dla_masked_spectrum_object, apply_recomended_corrections = True)
                corrected_o6_wavelength = non_dla_masked_spectrum_object.o6.lambdaa
                corrected_o6_redshift = non_dla_masked_spectrum_object.o6.z
                corrected_o6_tau = non_dla_masked_spectrum_object.o6.tau_rec
            if compare_corrections or no_metal_corrections:
                recover_o6(non_dla_masked_spectrum_object, apply_recomended_corrections = False)
                uncorrected_o6_wavelength = non_dla_masked_spectrum_object.o6.lambdaa
                uncorrected_o6_redshift = non_dla_masked_spectrum_object.o6.z
                uncorrected_o6_tau = non_dla_masked_spectrum_object.o6.tau_rec

        if corrected_h1_tau is not None:
            x_data.append(corrected_h1_wavelength if wavelength else corrected_h1_redshift if redshift else corrected_h1_redshift * speed_of_light_kms)
            flux_data.append(np.exp(-10**corrected_h1_tau))
            labels.append("H I" if not compare_corrections else "H I (corrected)")
            colours.append("black")
            linestyles.append("-")
            alphas.append(1.0)
        if uncorrected_h1_tau is not None:
            x_data.append(uncorrected_h1_wavelength if wavelength else uncorrected_h1_redshift if redshift else uncorrected_h1_redshift * speed_of_light_kms)
            flux_data.append(np.exp(-10**uncorrected_h1_tau))
            labels.append("Ly $\\alpha$" if not compare_corrections else "H I (Ly $\\alpha$)")
            colours.append("black")
            linestyles.append("--" if compare_corrections else "-")
            alphas.append(1.0)
        if corrected_c4_tau is not None:
            x_data.append(corrected_c4_wavelength if wavelength else corrected_c4_redshift if redshift else corrected_c4_redshift * speed_of_light_kms)
            flux_data.append(np.exp(-10**corrected_c4_tau))
            labels.append("C IV" if not compare_corrections else "C IV (corrected)")
            colours.append("blue")
            linestyles.append("-")
            alphas.append(1.0)
        if uncorrected_c4_tau is not None:
            x_data.append(uncorrected_c4_wavelength if wavelength else uncorrected_c4_redshift if redshift else uncorrected_c4_redshift * speed_of_light_kms)
            flux_data.append(np.exp(-10**uncorrected_c4_tau))
            labels.append("C IV" if not compare_corrections else "C IV (raw)")
            colours.append("blue")
            linestyles.append("--" if compare_corrections else "-")
            alphas.append(1.0)
        if corrected_o6_tau is not None:
            x_data.append(corrected_o6_wavelength if wavelength else corrected_o6_redshift if redshift else corrected_o6_redshift * speed_of_light_kms)
            flux_data.append(np.exp(-10**corrected_o6_tau))
            labels.append("O VI" if not compare_corrections else "O VI (corrected)")
            colours.append("red")
            linestyles.append("-")
            alphas.append(1.0)
        if uncorrected_o6_tau is not None:
            x_data.append(uncorrected_o6_wavelength if wavelength else uncorrected_o6_redshift if redshift else uncorrected_o6_redshift * speed_of_light_kms)
            flux_data.append(np.exp(-10**uncorrected_o6_tau))
            labels.append("O VI" if not compare_corrections else "O VI (raw)")
            colours.append("red")
            linestyles.append("--" if compare_corrections else "-")
            alphas.append(1.0)

    else:
        if h1:
            x_data.append(data["Wavelength_Ang"][:])
            flux_data.append(np.exp(-selected_spectrum_data["h1/RedshiftSpaceOpticalDepthOfStrongestTransition"][:]))
            labels.append("H I")
            colours.append("black")
            linestyles.append("-")
            alphas.append(1.0)
        if c4:
            x_data.append(data["Wavelength_Ang"][:])
            flux_data.append(np.exp(-selected_spectrum_data["c4/RedshiftSpaceOpticalDepthOfStrongestTransition"][:]))
            labels.append("C IV")
            colours.append("blue")
            linestyles.append("-")
            alphas.append(1.0)
        if o6:
            x_data.append(data["Wavelength_Ang"][:])
            flux_data.append(np.exp(-selected_spectrum_data["o6/RedshiftSpaceOpticalDepthOfStrongestTransition"][:]))
            labels.append("O VI")
            colours.append("red")
            linestyles.append("-")
            alphas.append(1.0)

    # Plot results

    figure, axis = plotting.spectrum(
        pixel_x_values = x_data,
        pixel_fluxes = flux_data,
        spectrum_type = plotting.PlottedSpectrumType.wavelength if wavelength else plotting.PlottedSpectrumType.velocity if velocity else plotting.PlottedSpectrumType.redshift,
        dataset_labels = labels,
        dataset_colours = colours,
        dataset_linestyles = linestyles,
        dataset_alphas = alphas,
        x_min = x_min,
        x_max = x_max,
        title = title,
        show = output_file is None
    )

    # Decide what to do with the plot

    if output_file is not None:
        figure.savefig(output_file)

def main():
    script_wrapper = ScriptWrapper("pod-plot-specwizard-spectrum",
                                   "Christopher Rowe",
                                   "1.0.0",
                                   "26/05/2024",
                                   "Plots a spectrum from data in SpecWizard output files.",
                                   ["podpy-refocused"],
                                   [],
                                   [],
                                   [["datafile",                "i",    "SpecWizard output file. Defaults to a file in the current working directory named \"LongSpectrum.hdf5\".", False, False, None, "./LongSpectrum.hdf5"],
                                    ["raw",                     None,   "Plot the raw flux.\nOnly supported for --wavelength", False, True, None, None, ["velocity", "redshift"]],
                                    ["h1",                      None,   "Plot fluxes from H I.", False, True, None, None],
                                    ["c4",                      None,   "Plot fluxes from C IV.", False, True, None, None],
                                    ["o6",                      None,   "Plot fluxes from O VI.", False, True, None, None],
                                    ["target-spectrum",         "s",    "Index of the spectrum to plot.\nDefault is 0.", False, False, float, 0],
                                    ["wavelength",              "w",    "Use wavelength pixels.", False, True, None, None, ["velocity", "redshift"]],
                                    ["velocity",                None,    "Use velocity pixels.", False, True, None, None, ["raw", "wavelength", "redshift"]],
                                    ["redshift",                "r",    "Use redshift pixels.", False, True, None, None, ["raw", "wavelength", "velocity"]],
                                    ["x-min",                   None,   "Minimum log10-space value of tau_HI to consider.\nDefault is -1.0", False, False, float, None],
                                    ["x-max",                   None,   "Maximum log10-space value of tau_HI to consider.\nDefault is 2.5", False, False, float,  None],
                                    ["no-h1-corrections",       None,   "Make no alterations to saturated H I pixels.", False, True, None, None, ["compare-corrections", "use-RSODOST-field"]],
                                    ["no-metal-corrections",    None,   "Make no alterations to saturated metal pixels.", False, True, None, None, ["compare-corrections", "use-RSODOST-field"]],
                                    ["compare-corrections",     None,   "Plot comparison lines for all selected ions to show their un-corrected values.", False, True, None, None, ["no-h1-corrections", "no-metal-corrections", "use-RSODOST-field"]],
                                    ["use-RSODOST-field",       None,   "Get ion flux from the 'RedshiftSpaceOpticalDepthOfStrongestTransition' field.", False, True, None, None, ["no-h1-corrections", "no-metal-corrections", "compare-corrections"]],
                                    ["title",                   None,   "Optional plot title.", False, False, None, None],
                                    ["output-file",             "o",    "Name of the file to save plot as. Leave unset to show the plot in a matplotlib interactive window.", False, False, None, None]
                                   ])
    
    script_wrapper.run(__main)
