# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from QuasarCode import Console
from typing import Union, List, Tuple, Dict, Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ._tools import PlotObjects, require_plottable, showable
from .._TauBinned import BinnedOpticalDepthResults

class PlottedSpectrumType(Enum):
    wavelength = 0
    velocity = 1
    redshift = 2

@showable(kwarg_name = "show")
@require_plottable(combined_kwarg_name = "plot_objects", figure_creation_settings_kwarg_name = "figure_creation_kwargs", figure_kwarg_name = "figure", axis_kwarg_name = "axis")
def plot_spectrum(
                    pixel_x_values: Union[List[np.ndarray], np.ndarray],
                    pixel_fluxes: Union[List[np.ndarray], np.ndarray],
                    spectrum_type: PlottedSpectrumType,
                    dataset_labels: List[str],
                    dataset_colours: Union[List[str], None],
                    dataset_linestyles: Union[List[str], None],
                    dataset_alphas: Union[List[float], None],
                    x_min: Union[float, None] = None,
                    x_max: Union[float, None] = None,
                    y_min: Union[float, None] = None,
                    y_max: Union[float, None] = None,
                    title: Union[str, None] = None,
                    #figure: Union[Figure, None] = None,
                    #axis: Union[Axes, None] = None,
                    #figure_creation_kwargs: Union[Dict[str, Any], None] = None,
                    plot_objects: Union[PlotObjects, None] = None,
                    #show: bool = False,
                ) -> PlotObjects:
    """
    Plot a spectrum from raw data.
    """

    if not isinstance(pixel_x_values, list):
        pixel_x_values = [pixel_x_values]
    if not isinstance(pixel_fluxes, list):
        pixel_fluxes = [pixel_fluxes]

    if plot_objects is None or plot_objects.axis is None:
        raise RuntimeError("Internal error! Required plot axis object not generated.")

    min_x_value = min(pixel_x_values[0])
    max_x_value = max(pixel_x_values[0])
    for i in range(len(pixel_fluxes)):
        if len(pixel_x_values) > 1 and i > 0:
            min_x_value = min(min_x_value, min(pixel_x_values[i]))
            max_x_value = max(max_x_value, max(pixel_x_values[i]))
        plot_objects.axis.plot(
            pixel_x_values[i] if len(pixel_x_values) > 1 else pixel_x_values[0],
            pixel_fluxes[i],
            color = dataset_colours[i] if dataset_colours is not None else None,
            linestyle = dataset_linestyles[i] if dataset_linestyles is not None else None,
            alpha = dataset_alphas[i] if dataset_alphas is not None else None,
            label = dataset_labels[i]
        )

    xlims = (min_x_value if x_min is None else x_min, max_x_value if x_max is None else x_max)
    widest_xlims = (min(min_x_value, xlims[0]), max(max_x_value, xlims[1]))
    ylims = (-0.25, plt.ylim()[1])
    plot_objects.axis.add_patch(Rectangle((widest_xlims[0], -1.0), widest_xlims[1] - widest_xlims[0], 1.0, facecolor = "grey", alpha = 0.8))
    plot_objects.axis.add_patch(Rectangle((widest_xlims[0], 1.0), widest_xlims[1] - widest_xlims[0], 1.0, facecolor = "grey", alpha = 0.8))

    plot_objects.axis.set_xlim(xlims)
    plot_objects.axis.set_ylim((ylims[0] if y_min is None else y_min, ylims[1] if y_max is None else y_max))

    if title is not None:
        plot_objects.axis.set_title(title)
    if spectrum_type == PlottedSpectrumType.wavelength:
        x_label = "Wavelength (A)"
    elif spectrum_type == PlottedSpectrumType.velocity:
        x_label = "Velocity (km/s)"
    elif spectrum_type == PlottedSpectrumType.redshift:
        x_label = "z"
    plot_objects.axis.set_xlabel(x_label)
    plot_objects.axis.set_ylabel("Normalised Flux ( exp(-$\\tau_Z$) )")
    plot_objects.axis.legend()

    return plot_objects
