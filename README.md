

<img width="1040" alt="image" src="https://github.com/user-attachments/assets/553ce3cc-4697-461f-b167-609a3f4fd7d3" />




# PlotMint

A modern, professional Python visualization library with 30+ pre-built plotting functions for data science and analytics. Built on top of matplotlib, seaborn, and pandas, PlotMint makes it easy to create beautiful, publication-ready charts with minimal code.

## Features
- 30+ ready-to-use plotting functions (bar, scatter, pie, time series, density, heatmap, treemap, violin, and more)
- Consistent, professional color schemes and themes
- Input validation and user-friendly error messages
- Easy integration with pandas DataFrames
- Export plots to high-quality images

## Installation
```bash
pip install -r requirements.txt
```

## Usage Example
```python
import pandas as pd
from plotmint import plotmint_improved as plt

df = pd.read_csv('your_data.csv')
plt.vis_bar(df, col='category_column', title='Bar Chart Example')
```

## Available Plots
- Bar, Ordered Bar, Horizontal Bar
- Scatter, Scatter with Hist/Box
- Pie, Pie with Explode
- Time Series, Seasonal, Timeline, Calendar Heatmap
- Density, KDE, Violin, Population Pyramid
- Treemap, Lollipop, Dot, Jitter, Divergence, Count, Categorical

## Professional Themes
All plots use a modern, professional color palette and style by default. You can further customize themes in `plotmint/utils.py`.

## License
MIT License

## Author
Mohd Azam 
