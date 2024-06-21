# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from .._Spectrum import from_SpecWizard, fit_continuum, recover_c4, recover_o6
from .._TauBinned import bin_combined_pixels_from_SpecWizard, BinnedOpticalDepthResults
from .. import plotting

from QuasarCode import Settings, Console
from QuasarCode.Tools import ScriptWrapper
from typing import Union, List
from matplotlib import pyplot as plt
import datetime
import os

DEFAULT_SPECWIZARD_FILENAME = "LongSpectrum.hdf5"



def __main(
            datafiles: List[str],
            combine_datafiles: bool,
            c4: bool, o6: bool,
            x_min: float, x_max: float, y_min: Union[float, None], y_max: Union[float, None],
            output_file: Union[str, None],
            input_los_mask: Union[List[List[bool]], None],
            aguirre05_obs: bool, turner16_obs: bool, turner16_synthetic: bool,
            flag_h1_corrections_only: bool, no_h1_corrections: bool, no_metal_corrections: bool,
            metal_flat_level_offset: Union[List[float], None],
            labels: Union[List[str], None],
            linestyles: Union[List[str], None],
            colours: Union[List[str], None],
            title: Union[str, None],
            density: bool,
            density_dataset: int,
            density_colourmap: str
        ) -> None:
    
    if not (c4 or o6):
        Console.print_error("Metal ion must be specified!")
        Console.print_info("Terminating...")
        return
    ion_string = "c4" if c4 else "o6"

    # Load and process data

    dataset_results = []
    if combine_datafiles:
        datasets = []
    for i in range(len(datafiles)):

        datafile = datafiles[i]
        
        if not os.path.isfile(datafile):
            if os.path.isdir(datafile):
                test_datafile = os.path.join(datafile, DEFAULT_SPECWIZARD_FILENAME)
                if os.path.isfile(test_datafile):
                    datafile = test_datafile
                else:
                    raise FileNotFoundError(f"Directory was provided, but contained no file named \"{DEFAULT_SPECWIZARD_FILENAME}\".\nDirectory was: {datafile}")
            else:
                raise FileNotFoundError(f"No file exists at: {datafile}")

        # Load the data (and recover H I)
        spectrum_objects = from_SpecWizard(
                                filepath = datafile,
                                sightline_filter = input_los_mask[i] if input_los_mask is not None else None,
                                identify_h1_contamination = not no_h1_corrections,
                                correct_h1_contamination = (not flag_h1_corrections_only) and (not no_h1_corrections),
                                mask_dla = c4
                        )

        # Recover metal optical depth
        if c4:
            fit_continuum(*spectrum_objects)
            recover_c4(*spectrum_objects, observed_log10_flat_level = None if metal_flat_level_offset is None else metal_flat_level_offset[i] if len(metal_flat_level_offset) > 1 else metal_flat_level_offset[0], apply_recomended_corrections = not no_metal_corrections)
        elif o6:
            recover_o6(*spectrum_objects, observed_log10_flat_level = None if metal_flat_level_offset is None else metal_flat_level_offset[i] if len(metal_flat_level_offset) > 1 else metal_flat_level_offset[0], apply_recomended_corrections = not no_metal_corrections)
        else:
            raise RuntimeError()

        if not combine_datafiles:
            # Combine and bin pixels
            dataset_results.append(bin_combined_pixels_from_SpecWizard("h1", ion_string, *spectrum_objects, x_limits = (x_min, x_max), n_bootstrap_resamples = 0, legacy = False))
        else:
            datasets.extend(spectrum_objects)

    if combine_datafiles:
        #TODO: check this is the right way to do this!
        dataset_results.append(bin_combined_pixels_from_SpecWizard("h1", ion_string, *datasets, x_limits = (x_min, x_max), n_bootstrap_resamples = 0, legacy = False))

    # Plot results

    figure, axis = plotting.pod_statistics(results = dataset_results[0], label = labels[0] if labels is not None else f"#{1}" if len(datafiles) > 1 and not combine_datafiles else "SpecWizard" if (aguirre05_obs or turner16_obs or turner16_synthetic) else None,
                                           colour = colours[0] if colours is not None else None, linestyle = linestyles[0] if linestyles is not None else None,
                                           x_min = x_min,
                                           x_max = x_max,
                                           y_min = y_min,
                                           y_max = y_max,
                                           title = title,
                                           x_label = "$\\rm log_{10}$ $\\tau_{\\rm H I}$",
                                           y_label = "Median $\\rm log_{10}$ $\\tau_{\\rm " + ("C IV" if c4 else "O VI") + "}$",
                                           density = density and (density_dataset == 0 or combine_datafiles),
                                           density_colourmap = density_colourmap,
                                           density_uses_all_pixels = True)#TODO: change this to be an option that defaults to False!
    if not combine_datafiles:
        for i in range(1, len(datafiles)):
            figure, axis = plotting.pod_statistics(results = dataset_results[i], label = labels[i] if labels is not None else f"#{i+1}",
                                                   colour = colours[i] if colours is not None else None, linestyle = linestyles[i] if linestyles is not None else None,
                                                   density = density and density_dataset == i,
                                                   density_colourmap = density_colourmap,
                                                   density_uses_all_pixels = True,#TODO: change this to be an option that defaults to False!
                                                   hide_tau_min_label = True, allow_auto_set_labels = False, figure = figure, axis = axis)
    #TODO: change these to the combined datasets or have flags/params for each dataset
    if aguirre05_obs:
        plotting.comparison_data.aguirre05.plot_obs(ion_string, label = "A+05 -- Obs. Data", axis = axis)
    if turner16_obs:
        plotting.comparison_data.turner16.plot_obs_Q1317_0507(ion_string, label = "T+16 -- Obs. Data", axis = axis)
    if turner16_synthetic:
        plotting.comparison_data.turner16.plot_synthetic_Q1317_0507(ion_string, label = "T+16 -- EAGLE", axis = axis)
    axis.legend()

    # Decide what to do with the plot

    if output_file is None:
        plt.show()
    else:
        figure.savefig(output_file)

def main():
    ScriptWrapper(
        command = "pod-plot-specwizard-relation",
        authors = [ScriptWrapper.AuthorInfomation(given_name = "Christopher", family_name = "Rowe", email = "contact@cjrrowe.com", website_url = "cjrrowe.com")],
        version = "1.3.0",
        edit_date = datetime.date(2024, 5, 28),
        description = "Plots the relation between two ion's tau values.",
        dependancies = ["podpy-refocused"],
        usage_paramiter_examples = None,
        parameters = [
            ScriptWrapper.OptionalParam[List[str]](
                "datafiles", "i",
                description = "Semicolon seperated list of SpecWizard output files.\nDefaults to a single file in the current working directory named \"LongSpectrum.hdf5\".",
                default_value = ["./LongSpectrum.hdf5"],
                conversion_function = ScriptWrapper.make_list_converter(";")
            ),
            ScriptWrapper.Flag(
                "combine-datafiles",
                description = "Combine the results from all data files into a single dataset.\nUses the method from Turner et al. 2016 for combining data from different QSOs."
            ),
            ScriptWrapper.Flag(
                "c4",
                description = "Select C IV as the metal ion.",
                conflicts = ["o6"]
            ),
            ScriptWrapper.Flag(
                "o6",
                description = "Select O VI as the metal ion.",
                conflicts = ["c4"]
            ),
            ScriptWrapper.OptionalParam[float](
                "x-min",
                description = "Minimum log10-space value of tau_HI to consider.\nDefault is -1.0",
                default_value = -1.0,
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float](
                "x-max",
                description = "Maximum log10-space value of tau_HI to consider.\nDefault is 2.5",
                default_value = 2.5,
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "y-min",
                description = "Minimum log10-space value of tau_Z to display.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "y-max",
                description = "Maximum log10-space value of tau_Z to display.\nNo default value, however 0.0 is a good choice for C IV.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[str | None](
                "output-file", "o",
                description = "Name of the file to save plot as. Leave unset to show the plot in a matplotlib interactive window."
            ),
            ScriptWrapper.OptionalParam[List[List[bool]] | None](
                "input-los-mask",
                description = "Mask for line-of-sight selection from input file(s).\nUse a string of 'T' and 'F' seperated by a semicolon for each input file if more than one is specified.\nFor example: \"TFTTF\" for a file with 5 lines of sight.\nSpecifying only one input mask and many files will result in the mask being appled to each file.\nSpecifying fewer mask elements than lines of sight avalible will result in just the first N lines of sight\nbeing considered for a mask of length N.",
                conversion_function = ScriptWrapper.make_list_converter(";", lambda v: [e.lower() == "t" for e in v])
            ),
            ScriptWrapper.Flag(
                "aguirre05-obs",
                description = "Plot the observational data from Aguirre et al. 2005."
            ),
            ScriptWrapper.Flag(
                "turner16-obs",
                description = "Plot the observational data from Turner et al. 2016."
            ),
            ScriptWrapper.Flag(
                "turner16-synthetic",
                description = "Plot the SpecWizard data from Turner et al. 2016."
            ),
            ScriptWrapper.Flag(
                "flag-h1-corrections-only",
                description = "Identify saturated H I pixels, but make no attempt to correct them."
            ),
            ScriptWrapper.Flag(
                "no-h1-corrections",
                description = "Make no alterations to saturated H I pixels."
            ),
            ScriptWrapper.Flag(
                "no-metal-corrections",
                description = "Make no alterations to saturated metal pixels."
            ),
            ScriptWrapper.OptionalParam[List[float] | None](
                "metal-flat-level-offset",
                description = "Value of log10 tau_Z to add to recovered metal optical depths (in non-log space).\nUse a semicolon seperated list if a different value is required for each data file.",
                conversion_function = ScriptWrapper.make_list_converter(";")
            ),
            ScriptWrapper.OptionalParam[List[str] | None](
                "labels",
                description = "Custom dataset label(s).\nUse a semicolon seperated list if more than one data file is specified.",
                conversion_function = ScriptWrapper.make_list_converter(";")
            ),
            ScriptWrapper.OptionalParam[List[str] | None](
                "linestyles",
                description = "Custom dataset line style(s).\nUse a semicolon seperated list if more than one data file is specified.",
                conversion_function = ScriptWrapper.make_list_converter(";")
            ),
            ScriptWrapper.OptionalParam[List[str] | None](
                "colours",
                description = "Custom dataset line colour(s).\nUse a semicolon seperated list if more than one data file is specified.",
                conversion_function = ScriptWrapper.make_list_converter(";")
            ),
            ScriptWrapper.OptionalParam[str | None](
                "title",
                description = "Optional plot title."
            ),
            ScriptWrapper.Flag(
                "density",
                description = "Show a density hexplot of the pixel pairs."
            ),
            ScriptWrapper.OptionalParam[int](
                "density-dataset",
                description = "Index of the dataset to use for the density hexplot when more than one non-combined dataset is provided. Defaults to index 0.",
                default_value = 0,
                conversion_function = int
            ),
            ScriptWrapper.OptionalParam[str](
                "density-colourmap",
                description = "Colourmap to use for the density hexplot.\nDefault is viridis.",
                default_value = "viridis"
            ),
        ]
    ).run(__main)
