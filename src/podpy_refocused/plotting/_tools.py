# SPDX-FileCopyrightText: 2024-present Christopher J. R. Rowe <contact@cjrrowe.com>
#
# SPDX-License-Identifier: MIT
from QuasarCode import Console
from matplotlib import pyplot as plt
from functools import wraps
from collections import namedtuple
from typing import Union, Tuple, Dict, Any

from matplotlib.figure import Figure
from matplotlib.axes import Axes

PlotObjects = namedtuple(typename = "PlotObjects",
                         field_names = ["figure", "axis"],
                         defaults = [None, None])

def ensure_plottable(plot_objects: PlotObjects = None,
                     figure_creation_kwargs: Union[Dict[str, Any], None] = None) -> Tuple[Union[Figure, None], Axes]:
#def ensure_plottable(figure: Union[Figure, None] = None,
#                     axis: Union[Axes, None] = None,
#                     figure_creation_kwargs: Union[Dict[str, Any], None] = None) -> Tuple[Union[Figure, None], Axes]:
    figure = plot_objects.figure
    axis = plot_objects.axis
    if figure is None:
        if axis is None:
            figure = plt.figure(**(figure_creation_kwargs if figure_creation_kwargs is not None else {}))
            axis = figure.gca()
    elif axis is None:
        axis = figure.gca()
    return PlotObjects(figure = figure, axis = axis)

def require_plottable(func):
    """
    Autogenerates a missing axis object from an exising figure, or a new figure & axis pair.
    Provides both objects as a PlotObjects named tuple (names are 'figure' and 'axis').
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        plot_objects = ensure_plottable(
            plot_objects = kwargs.pop("plot_objects", PlotObjects(figure = kwargs.pop("figure", None), axis = kwargs.pop("figure_creation_kwargs", None))),
            figure_creation_kwargs = kwargs.pop("figure_creation_kwargs", None)
        )
        return func(*args, **kwargs, plot_objects = plot_objects)
    return wrapper

def showable(func):
    """
    Allows a plot to be shown in an interactive window.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        show = kwargs.pop("show", False)
        result: PlotObjects = func(*args, **kwargs)
        if show:
            if isinstance(result, PlotObjects):
                if result.figure is None:
                    Console.print_warning("Unable to show figure as axis object reference provided without figure object reference.")
                else:
                    result.figure.show()
        return result
    return wrapper