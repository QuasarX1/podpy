# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from .._SpecWizard import SpecWizard_NoiseProfile
from .. import plotting

import datetime
import os
from typing import Union, List, Callable

from QuasarCode import Settings, Console
from QuasarCode.Tools import ScriptWrapper, Struct, TypedAutoProperty, TypeCastAutoProperty, TypeShield, NestedTypeShield, cast_unyt_quantity, cast_unyt_array
from matplotlib import pyplot as plt
import numpy as np
from unyt import unyt_quantity, unyt_array, angstrom
from astropy.io import fits
import h5py as h5



def main():
    ScriptWrapper(
        command = "pod-plot-specwizard-noise-file",
        authors = [ScriptWrapper.AuthorInfomation(given_name = "Christopher", family_name = "Rowe", email = "contact@cjrrowe.com", website_url = "cjrrowe.com")],
        version = "1.0.0",
        edit_date = datetime.date(2024, 5, 28),
        description = "Plots the relation between two ion's tau values.",
        dependancies = ["podpy-refocused"],
        usage_paramiter_examples = None,
        parameters = [
            ScriptWrapper.PositionalParam[str | None](
                name = "file",
                short_name = "i",
                description = "Target SpecWizard noise profile file."
            ),
            ScriptWrapper.OptionalParam[unyt_quantity | None](
                "min-wavelength",
                description = "Minimum wavelength (in Angstroms) to show.",
                conversion_function = lambda s: unyt_quantity(float(s), units = angstrom)
            ),
            ScriptWrapper.OptionalParam[unyt_quantity | None](
                "max-wavelength",
                description = "Maximum wavelength (in Angstroms) to show.",
                conversion_function = lambda s: unyt_quantity(float(s), units = angstrom)
            ),
            ScriptWrapper.Flag(
                "interpolate-wavelength",
                description = "Interpolate in wavelength to produce wavelength pixels with a size defined by --mapped-wavelength-pixel-size."
            ),
            ScriptWrapper.OptionalParam[unyt_quantity](
                "mapped-wavelength-pixel-size",
                description = "When wavelength interpolation is enabled, interpolate onto pixels with this width in Angstroms.\nDefaults to 0.1.",
                default_value = 0.1,
                conversion_function = lambda s: unyt_quantity(float(s), units = angstrom)
            ),
            ScriptWrapper.OptionalParam[float | None](
                "min-flux",
                description = "Minimum normalised flux to show.",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "max-flux",
                description = "Maximum normalised flux to show.",
                conversion_function = float
            ),
            ScriptWrapper.Flag(
                "interpolate-flux",
                description = "Interpolate in flux to produce flux pixels with a size defined by --mapped-flux-pixel-size."
            ),
            ScriptWrapper.OptionalParam[float](
                "mapped-flux-pixel-size",
                description = "When flux interpolation is enabled, interpolate onto pixels with this width in normalised flux.\nDefaults to 0.01.",
                default_value = 0.01,
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "min-sigma",
                description = "Minimum sigma value to show.\nSmaller values will be set to this value (including zero values).",
                conversion_function = float
            ),
            ScriptWrapper.OptionalParam[float | None](
                "max-sigma",
                description = "Maximum sigma value to show.\nLarger values will be restricted to this value.",
                conversion_function = float,
            ),
            ScriptWrapper.OptionalParam[str | None](
                name = "plot-filename",
                short_name = "p",
                description = "Plot the results and save the image to this file."
            ),
            ScriptWrapper.Flag(
                name = "show",
                short_name = "s",
                sets_param = "show_plot",
                description = "Plot the results and show using the matplotlib interactive window."
            ),
            ScriptWrapper.OptionalParam[str](
                name = "colourmap-name",
                short_name = "c",
                description = "Set the colourmap used by matplotlib using a built-in colourmap name.\nDefaults to \"viridis\".",
                default_value = "viridis"
            ),
            ScriptWrapper.OptionalParam[float](
                name = "figure-width",
                description = "Set the width of the plot in inches.\nDefaults to 12.",
                default_value = 12.0
            ),
            ScriptWrapper.OptionalParam[float](
                name = "figure-height",
                description = "Set the height of the plot in inches.\nDefaults to 6.",
                default_value = 6.0
            )
        ]
    ).run(__main)




def __main(
            file: str,
            min_wavelength: Union[unyt_quantity, None], # Angstroms
            max_wavelength: Union[unyt_quantity, None], # Angstroms
            interpolate_wavelength: bool,
            mapped_wavelength_pixel_size: unyt_quantity, # Angstroms
            min_flux: Union[float, None], # Normalised Flux
            max_flux: Union[float, None], # Normalised Flux
            interpolate_flux: bool,
            mapped_flux_pixel_size: float, # Normalised Flux
            min_sigma: Union[float, None],
            max_sigma: Union[float, None],
            plot_filename: str,
            show_plot: bool,
            colourmap_name: str,
            figure_width: float,
            figure_height: float
        ) -> None:

    if not show_plot and plot_filename is None:
        Console.print_error("No output format(s) specified. At least one of --show and --plot-filename must be specified.")
        return

    # Read file

#    with h5.File(file, "r") as f:
#        flux_centres = f["NormalizedFlux"][:]
#        selected_wavelengths = unyt_array(f["Wavelength_Angstrom"][:], units = angstrom)
#        sigma_of_flux = f["NormalizedNoise"][:].T
    noise_profile = SpecWizard_NoiseProfile.read(file)

    # Plot

    fig, axis = plotting.specwizard_noise_table(
#        wavelengths = selected_wavelengths,
#        flux_bin_centres = flux_centres,
#        sigma_table = sigma_of_flux,
        noise_profile = noise_profile,
        wavelength_interpolated_pixel_size = mapped_wavelength_pixel_size if interpolate_wavelength else None,
        flux_interpolated_pixel_size = mapped_flux_pixel_size if interpolate_flux else None,
        wavelength_axis_limits = (min_wavelength, max_wavelength),
        flux_axis_limits = (min_flux, max_flux),
        sigma_limits = (min_sigma, max_sigma),
        colourmap = colourmap_name,
        axis_label_fontsize = "large",
        figure_creation_kwargs = {
            "figsize": (figure_width, figure_height),
            "layout": "constrained"
#                "layout": "tight"
        }
    )

    # Save plot

    if plot_filename is not None:
        fig.savefig(plot_filename)

    # Show plot

    if show_plot:
        plt.show()
