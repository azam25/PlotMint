"""
Utility functions for the PlotMint library.

This module contains helper functions for data validation, configuration management,
and common operations used across the plotting functions.

Author: Mohd Azam
Date: 2024-06-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple, Any
import warnings


class PlotConfig:
    """Global configuration for all plots in PlotMint."""
    
    def __init__(self):
        self.default_figsize = (12, 8)
        self.default_fontsize = 12
        self.default_title_fontsize = 16
        self.default_label_fontsize = 12
        self.default_tick_fontsize = 10
        self.color_palette = 'viridis'
        self.style = 'seaborn'
        self.dpi = 80
        self.facecolor = 'white'
        self.edgecolor = 'black'
    
    def apply_defaults(self):
        """Apply default matplotlib settings."""
        plt.style.use(self.style)
        plt.rcParams['figure.figsize'] = self.default_figsize
        plt.rcParams['font.size'] = self.default_fontsize
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['figure.facecolor'] = self.facecolor
        plt.rcParams['figure.edgecolor'] = self.edgecolor


class DataValidator:
    """Validate input data for plotting functions."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """
        Validate that the input is a pandas DataFrame and contains required columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe to validate
        required_columns : List[str], optional
            List of column names that must be present in the dataframe
            
        Returns:
        --------
        bool
            True if validation passes, raises ValueError otherwise
            
        Raises:
        -------
        ValueError
            If df is not a DataFrame or missing required columns
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        return True
    
    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> bool:
        """
        Validate that specified columns contain numeric data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            List of column names to validate
            
        Returns:
        --------
        bool
            True if all columns are numeric, raises ValueError otherwise
        """
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric")
        
        return True
    
    @staticmethod
    def validate_date_column(df: pd.DataFrame, date_col: str) -> bool:
        """
        Validate that a column contains date-like data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        date_col : str
            Name of the date column
            
        Returns:
        --------
        bool
            True if column contains date-like data, raises ValueError otherwise
        """
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        
        # Try to convert to datetime
        try:
            pd.to_datetime(df[date_col])
        except (ValueError, TypeError):
            raise ValueError(f"Column '{date_col}' must contain date-like data")
        
        return True


class PlotTheme:
    """Manage plot styling and themes."""
    
    THEMES = {
        'default': {
            'style': 'default',
            'palette': 'viridis',
            'figsize': (12, 8),
            'fontsize': 12,
            'title_fontsize': 16,
            'label_fontsize': 12,
            'tick_fontsize': 10
        },
        'dark': {
            'style': 'dark_background',
            'palette': 'plasma',
            'figsize': (12, 8),
            'fontsize': 12,
            'title_fontsize': 16,
            'label_fontsize': 12,
            'tick_fontsize': 10
        },
        'minimal': {
            'style': 'default',
            'palette': 'gray',
            'figsize': (10, 6),
            'fontsize': 10,
            'title_fontsize': 14,
            'label_fontsize': 10,
            'tick_fontsize': 8
        },
        'professional': {
            'style': 'default',
            'palette': 'Set2',
            'figsize': (12, 8),
            'fontsize': 11,
            'title_fontsize': 14,
            'label_fontsize': 11,
            'tick_fontsize': 9
        }
    }
    
    @staticmethod
    def apply_theme(theme_name: str = 'default') -> dict:
        """
        Apply a predefined theme to matplotlib.
        
        Parameters:
        -----------
        theme_name : str
            Name of the theme to apply ('default', 'dark', 'minimal', 'professional')
            
        Returns:
        --------
        dict
            Theme configuration dictionary
        """
        if theme_name not in PlotTheme.THEMES:
            warnings.warn(f"Theme '{theme_name}' not found, using 'default'")
            theme_name = 'default'
        
        theme = PlotTheme.THEMES[theme_name]
        plt.style.use(theme['style'])
        
        return theme
    
    @staticmethod
    def get_color_palette(palette_name: str, n_colors: int = 10) -> List[str]:
        """
        Get a color palette with specified number of colors.
        
        Parameters:
        -----------
        palette_name : str
            Name of the color palette
        n_colors : int
            Number of colors to generate
            
        Returns:
        --------
        List[str]
            List of color hex codes
        """
        try:
            import seaborn as sns
            return sns.color_palette(palette_name, n_colors).as_hex()
        except ImportError:
            # Fallback to matplotlib colors
            cmap = plt.cm.get_cmap(palette_name)
            return [cmap(i) for i in np.linspace(0, 1, n_colors)]


class PlotExporter:
    """Handle plot export in multiple formats."""
    
    SUPPORTED_FORMATS = ['png', 'pdf', 'svg', 'jpg', 'jpeg', 'tiff']
    
    @staticmethod
    def save_plot(fig: plt.Figure, path: str, format: str = 'png', 
                  dpi: int = 300, bbox_inches: str = 'tight', 
                  facecolor: str = 'white', edgecolor: str = 'none',
                  **kwargs) -> None:
        """
        Save plot in multiple formats with consistent settings.
        
        Parameters:
        -----------
        fig : plt.Figure
            Matplotlib figure to save
        path : str
            File path to save the plot
        format : str
            Output format ('png', 'pdf', 'svg', 'jpg', 'jpeg', 'tiff')
        dpi : int
            Resolution for raster formats
        bbox_inches : str
            Bounding box setting
        facecolor : str
            Figure face color
        edgecolor : str
            Figure edge color
        **kwargs
            Additional arguments passed to fig.savefig()
        """
        if format.lower() not in PlotExporter.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. "
                           f"Supported formats: {PlotExporter.SUPPORTED_FORMATS}")
        
        # Ensure path has correct extension
        if not path.lower().endswith(f'.{format.lower()}'):
            path = f"{path}.{format.lower()}"
        
        # Save the plot
        fig.savefig(
            path,
            format=format,
            dpi=dpi,
            bbox_inches=bbox_inches,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs
        )


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Parameters:
    -----------
    filename : str
        Original filename
        
    Returns:
    --------
    str
        Sanitized filename
    """
    import re
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    return filename


def get_figure_size(width: Union[int, float] = None, 
                   height: Union[int, float] = None,
                   default: Tuple[int, int] = (12, 8)) -> Tuple[int, int]:
    """
    Get figure size with fallback to defaults.
    
    Parameters:
    -----------
    width : Union[int, float], optional
        Figure width
    height : Union[int, float], optional
        Figure height
    default : Tuple[int, int]
        Default figure size if width/height not provided
        
    Returns:
    --------
    Tuple[int, int]
        Figure size (width, height)
    """
    if width is None and height is None:
        return default
    elif width is None:
        return (default[0], height)
    elif height is None:
        return (width, default[1])
    else:
        return (width, height)


def validate_fontsize(fontsize: Union[int, float], 
                     min_size: int = 6, 
                     max_size: int = 72) -> Union[int, float]:
    """
    Validate and clamp font size to reasonable bounds.
    
    Parameters:
    -----------
    fontsize : Union[int, float]
        Input font size
    min_size : int
        Minimum allowed font size
    max_size : int
        Maximum allowed font size
        
    Returns:
    --------
    Union[int, float]
        Validated font size
    """
    if fontsize < min_size:
        warnings.warn(f"Font size {fontsize} is too small, using {min_size}")
        return min_size
    elif fontsize > max_size:
        warnings.warn(f"Font size {fontsize} is too large, using {max_size}")
        return max_size
    return fontsize


# Global configuration instance
config = PlotConfig() 