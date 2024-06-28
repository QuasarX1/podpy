# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from ._tools import PlotObjects, require_plottable, showable
from .._SpecWizard import SpecWizard_NoiseProfile

from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from QuasarCode import Console
from typing import Union, List, Tuple, Dict, Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from unyt import unyt_quantity, unyt_array, angstrom
from scipy.interpolate import RegularGridInterpolator

@showable(kwarg_name = "show")
@require_plottable(combined_kwarg_name = "plot_objects", figure_creation_settings_kwarg_name = "figure_creation_kwargs", figure_kwarg_name = "figure", axis_kwarg_name = "axis")
def plot_noise_table(
#                    wavelengths: unyt_array,
#                    flux_bin_centres: np.ndarray,
#                    sigma_table: np.ndarray,
                    noise_profile: SpecWizard_NoiseProfile,
                    sigma_label: Union[str, None] = None,
                    wavelength_interpolated_pixel_size: Union[unyt_quantity, None] = None,
                    flux_interpolated_pixel_size: Union[float, None] = None,
                    title: Union[str, None] = None,
                    wavelength_axis_limits: Union[Tuple[Union[unyt_quantity, None], Union[unyt_quantity, None]], None] = None,
                    flux_axis_limits: Union[Tuple[Union[float, None], Union[float, None]], None] = None,
                    sigma_limits: Union[Tuple[Union[float, None], Union[float, None]], None] = None,
                    colourmap: Union[str, None] = None,
                    axis_label_fontsize: Union[int, str] = "medium",
                    title_fontsize: Union[int, str] = "large",
                    #figure: Union[Figure, None] = None,
                    #axis: Union[Axes, None] = None,
                    #figure_creation_kwargs: Union[Dict[str, Any], None] = None,
                    plot_objects: Union[PlotObjects, None] = None,
                    #show: bool = False,
                ) -> PlotObjects:
    """
    Plot data from a specwizard noise file.

    Font size strings must be one of:
        xx-small
        x-small
        small
        medium
        large
        x-large
        xx-large
    """

    if plot_objects is None or plot_objects.figure is None:
        raise RuntimeError("Internal error! No figure generated or passed.")
    if plot_objects is None or plot_objects.axis is None:
        raise RuntimeError("Internal error! Required plot axis object not generated.")
    
#    if wavelength_interpolated_pixel_size is not None or flux_interpolated_pixel_size is not None:
#        sigma_table = RegularGridInterpolator((wavelengths, flux_bin_centres), sigma_table)(
#            np.meshgrid(
#                wavelengths if wavelength_interpolated_pixel_size is None else np.linspace(wavelengths[0], wavelengths[-1], int((wavelengths[-1] - wavelengths[0]) / wavelength_interpolated_pixel_size)),
#                flux_bin_centres if flux_interpolated_pixel_size is None else np.linspace(flux_bin_centres[0], flux_bin_centres[-1], int((flux_bin_centres[-1] - flux_bin_centres[0]) / flux_interpolated_pixel_size)),
#                indexing = "ij"
#            )
#        )
    if wavelength_interpolated_pixel_size is not None or flux_interpolated_pixel_size is not None:
        sigma_table = noise_profile.interpolator(
            np.meshgrid(
                noise_profile.wavelengths if wavelength_interpolated_pixel_size is None else np.linspace(noise_profile.wavelengths[0], noise_profile.wavelengths[-1], int((noise_profile.wavelengths[-1] - noise_profile.wavelengths[0]) / wavelength_interpolated_pixel_size)),
                noise_profile.normalised_fluxes if flux_interpolated_pixel_size is None else np.linspace(noise_profile.normalised_fluxes[0], noise_profile.normalised_fluxes[-1], int((noise_profile.normalised_fluxes[-1] - noise_profile.normalised_fluxes[0]) / flux_interpolated_pixel_size)),
                indexing = "ij"
            )
        )
    else:
        sigma_table = noise_profile.sigma_table

    if sigma_limits is not None:
        if sigma_limits[0] is not None:
            sigma_table[sigma_table < sigma_limits[0]] = sigma_limits[0]
        if sigma_limits[1] is not None:
            sigma_table[sigma_table > sigma_limits[1]] = sigma_limits[1]
    
    image_plot_object = plot_objects.axis.imshow(sigma_table.T, aspect = "auto", origin = "lower", extent = (noise_profile.wavelengths[0].value, noise_profile.wavelengths[-1].value, noise_profile.normalised_fluxes[0], noise_profile.normalised_fluxes[-1]), interpolation = "none", cmap = colourmap)
    plot_objects.figure.colorbar(image_plot_object, ax = plot_objects.axis).set_label(label = sigma_label if sigma_label is not None else "$\\sigma$", size = axis_label_fontsize)
    plot_objects.axis.set_xlim((noise_profile.wavelengths[0].value if wavelength_axis_limits is None or wavelength_axis_limits[0] is None else wavelength_axis_limits[0].value, noise_profile.wavelengths[-1].value if wavelength_axis_limits is None or wavelength_axis_limits[1] is None else wavelength_axis_limits[1].value))
    plot_objects.axis.set_ylim((noise_profile.normalised_fluxes[0] if flux_axis_limits is None or flux_axis_limits[0] is None else flux_axis_limits[0], noise_profile.normalised_fluxes[-1] if flux_axis_limits is None or flux_axis_limits[1] is None else flux_axis_limits[1]))
    plot_objects.axis.set_xlabel(f"Wavelength ({angstrom.units})", fontsize = axis_label_fontsize)
    plot_objects.axis.set_ylabel("Normalized Flux", fontsize = axis_label_fontsize)
    if title is not None:
        plot_objects.axis.set_title(title, fontsize = title_fontsize)

    return plot_objects
