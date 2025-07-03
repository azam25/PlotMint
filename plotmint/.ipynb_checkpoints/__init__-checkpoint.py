"""
PlotMint - A comprehensive Python visualization library

A collection of 30+ pre-built plotting functions for data visualization,
built on top of matplotlib, seaborn, and other visualization libraries.

Author: Mohd Azam
Date: 2024-06-13
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Mohd Azam"

# Import all visualization functions
from .plotmint import (
    # Basic plots
    vis_bar,
    vis_orderedBar,
    vis_scatter,
    vis_piechart,
    vis_piechartWithExplode,
    
    # Time series plots
    vis_seasonalPlot,
    vis_timeline,
    vis_timelineWithPeaksAndThroughs,
    vis_timeseries,
    vis_calendarHeatMap,
    
    # Statistical plots
    vis_density_curve,
    vis_densityPlot,
    vis_heatmap,
    vis_crossCorrelation,
    vis_divergence,
    vis_divergence_bar,
    
    # Advanced plots
    vis_countPlot,
    vis_lollipop,
    vis_populationPyramid,
    vis_treemap,
    vis_violen,
    vis_jitteringPlot,
    
    # Area and specialized plots
    vis_stackedAreaPlot,
    vis_unstackedAreaPlot,
    vis_scatter_with_hist,
    vis_scatter_with_box,
    
    # Utility functions
    stacked_hist,
    horizontalBargraph,
    vis_dotPlot,
    vis_categoricalPlot,
)

# Define what gets imported with "from plotmint import *"
__all__ = [
    # Basic plots
    "vis_bar",
    "vis_orderedBar", 
    "vis_scatter",
    "vis_piechart",
    "vis_piechartWithExplode",
    
    # Time series plots
    "vis_seasonalPlot",
    "vis_timeline",
    "vis_timelineWithPeaksAndThroughs",
    "vis_timeseries",
    "vis_calendarHeatMap",
    
    # Statistical plots
    "vis_density_curve",
    "vis_densityPlot",
    "vis_heatmap",
    "vis_crossCorrelation",
    "vis_divergence",
    "vis_divergence_bar",
    
    # Advanced plots
    "vis_countPlot",
    "vis_lollipop",
    "vis_populationPyramid",
    "vis_treemap",
    "vis_violen",
    "vis_jitteringPlot",
    
    # Area and specialized plots
    "vis_stackedAreaPlot",
    "vis_unstackedAreaPlot",
    "vis_scatter_with_hist",
    "vis_scatter_with_box",
    
    # Utility functions
    "stacked_hist",
    "horizontalBargraph",
    "vis_dotPlot",
    "vis_categoricalPlot",
]

# For improved API
from .plotmint_improved import * 