import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import squarify 
import calmap
from .utils import PlotTheme

# Helper to get professional palette
get_palette = lambda n: PlotTheme.get_color_palette('Set2', n)

def vis_seasonalPlot(df, date_col, value_col, fig_width = 12, fig_height = 6, title = '', title_fontsize = 16,
                     xlabel='', ylabel = '', xticks_fontsize = 12, yticks_fontsize = 12, legend_fontsize = 12 ,
                     xlabel_fonsize = 12, ylabel_fonsize = 12, label='', fileName = ''):
    """
    Module to help us analyze the historical data by giving an overview about the flow of a particular feature on an yearly basis.
    This plot can come handy in order to understand the seasonal trend.
    """
    # Prepare data
    df[date_col + '_year'] = [parse(d).year for d in df[date_col]]
    df[date_col + '_month'] = [parse(d).strftime('%b') for d in df[date_col]]
    years = df[date_col + '_year'].unique()

    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    colors = get_palette(len(years))
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)

    for i, y in enumerate(years):
        plt.plot(date_col + '_month', value_col, data=df.loc[df[date_col + '_year']==y, :], color=colors[i], label=y)
        plt.text(df.loc[df[date_col + '_year']==y, :].shape[0]-.9, df.loc[df[date_col + '_year']==y,value_col][-1:].values[0], y, fontsize=12, color=colors[i])
        
    # Decoration
    plt.ylim(50,750)
    plt.xlim(-0.3, 11)
    plt.xlabel(xlabel, fontsize=xlabel_fonsize)
    plt.ylabel(ylabel, fontsize=ylabel_fonsize)
    plt.xticks(fontsize=xticks_fontsize, alpha=.7)
    plt.yticks(fontsize=yticks_fontsize, alpha=.7)
    plt.title(title, fontsize=title_fontsize)
    plt.grid(axis='y', alpha=.3)

    # Remove borders
    plt.gca().spines["top"].set_alpha(0.0)    
    plt.gca().spines["bottom"].set_alpha(0.5)
    plt.gca().spines["right"].set_alpha(0.0)    
    plt.gca().spines["left"].set_alpha(0.5)   
    # plt.legend(loc='upper right', ncol=2, fontsize=12)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
def vis_scatter(df,category_col, x_axis_col, y_axis_col, fig_width = 16, fig_height = 10, scatter_size = 20,
                xlabel = '', ylabel='',title = '', title_font_size = 22, xtick_fontsize = 12, ytick_fontsize=12,
                xlabel_fontsize = 12, ylabel_fontsize = 12,  legend_fontsize = 12, fileName = ''):
    """
    Module to visualise multiple group data with different colors to study the relationship between two feature values. 
    """
    
    # Create as many colors as there are unique df['category']
    categories = np.unique(df[category_col])

    # Draw Plot for Each Category
    theme_config = PlotTheme.apply_theme('professional')
    colors = get_palette(len(categories))
    plt.figure(figsize=(fig_width, fig_height), dpi= 80, facecolor='w', edgecolor='k')

    for i, category in enumerate(categories):
        plt.scatter(x_axis_col, y_axis_col, 
                    data=df.loc[df.category==category, :], 
                    s=20, label=str(category), color=colors[i])

    plt.xlabel(xlabel,fontsize=xlabel_fontsize)
    plt.ylabel(ylabel,fontsize=ylabel_fontsize)
    
    plt.xticks(fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.title(title, fontsize=title_font_size)
    plt.legend(fontsize=legend_fontsize)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
def vis_divergence(df, diverging_col, y_axix_col, fig_width = 14, fig_height = 10, text_size = 10, yticks_font_size = 12, fig_title = '', title_text_size = 20,
                   xtick_min = -2.5, xtick_max =2.5, xtick_fontsize = 12, xlabel = '', ylabel = '',  xlabel_fontsize = 12, ylabel_fontsize = 12, fileName = ''):
    
    """
    This module helps us to visualize how the items are varying based on a single metric and analyze the order and amount of this variance.
    """
    
    x = df.loc[:, [diverging_col]]
    # Creating a new column having standard deviation of the diverging column
    df[diverging_col + '_z'] = (x - x.mean())/x.std()
    
    # Using up different color codes to distinguish between the various classes/categories present
    df['colors'] = ['red' if x < 0 else 'green' for x in df[diverging_col + '_z']]
    df.sort_values(diverging_col + '_z', inplace=True)
    df.reset_index(inplace=True)

    # Draw plot
    theme_config = PlotTheme.apply_theme('professional')
    colors = get_palette(2)
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z, color=colors, alpha=0.4, linewidth=5)

    # Decorations    
    plt.yticks(df.index, df[y_axix_col], fontsize=yticks_font_size)
    plt.xlabel(xlabel,fontsize=xlabel_fontsize)
    plt.ylabel(ylabel,fontsize=ylabel_fontsize)
    plt.title(fig_title, fontdict={'size':title_text_size})
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.xticks(fontsize=xtick_fontsize)
    plt.tight_layout()
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
    
def vis_divergence_bar(df, diverging_col, y_axix_col, fig_width = 14, fig_height = 10, text_size = 10, yticks_font_size = 12, fig_title = '', title_text_size = 20,
                   xtick_min = -2.5, xtick_max =2.5, xtick_fontsize = 12, xlabel = '', ylabel = '',  xlabel_fontsize = 12, ylabel_fontsize = 12, fileName = ''):
    
    """
    This module helps us to visualize how the items are varying based on a single metric and analyze the order and amount of this variance.
    """
    
    x = df.loc[:, [diverging_col]]
    # Creating a new column having standard deviation of the diverging column
    df[diverging_col + '_z'] = (x - x.mean())/x.std()
    
    # Using up different color codes to distinguish between the various classes/categories present
    df['colors'] = ['red' if x < 0 else 'green' for x in df[diverging_col + '_z']]
    df.sort_values(diverging_col + '_z', inplace=True)
    df.reset_index(inplace=True)

    # Draw plot
    theme_config = PlotTheme.apply_theme('professional')
    colors = get_palette(2)
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z, color=colors, alpha=0.4, linewidth=5)

    # Decorations    
    plt.yticks(df.index, df[y_axix_col], fontsize=yticks_font_size)
    plt.xlabel(xlabel,fontsize=xlabel_fontsize)
    plt.ylabel(ylabel,fontsize=ylabel_fontsize)
    plt.title(fig_title, fontdict={'size':title_text_size})
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-2.5, 2.5)
    plt.xticks(fontsize=xtick_fontsize)
    plt.tight_layout()
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    

def stacked_hist(df, x_col, group_col, fig_width, fig_height, bin_size, title = '', title_fontsize = 18, xlabel='',
                 ylabel='Frequency',xlabel_fontsize = 12, ylabel_fontsize = 12, fileName = '' , legend_fontsize = 10, xtick_fontsize = 10, ytick_fontsize = 10):
    """
    This module helps us create a Histogram that shows the frequency distribution of a given feature. 
    The idea behind this is to group the frequency bars based on a categorical variable
    to give a better insight about the continuous variable and the categorical variable in tandem.
    """
    # Prepare data
    df_agg = df.loc[:, [x_col, group_col]].groupby(group_col)
    vals = [df[x_col].values.tolist() for i, df in df_agg]

    # Plot figure
    theme_config = PlotTheme.apply_theme('professional')
    colors = get_palette(len(vals))
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    
    n, bins, patches = plt.hist(vals, bin_size ,stacked=True, density=False, color=colors[:len(vals)])

    # Decoration
    plt.legend({group:col for group, col in zip(np.unique(df[group_col]).tolist(), colors[:len(vals)])}, fontsize=legend_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize = xtick_fontsize)
    plt.yticks(fontsize = ytick_fontsize)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
    
def vis_piechart(df, groupby_col, fig_width=8, fig_height=8, title='', title_fontsize =14,annotation_textsize = 12,
                 fileName = ''):
    """
    Module to show the distribution of groups present in the particular feature using Pie chart
    """
    # Prepare Data
    df = df.groupby(groupby_col).size()

    # Make the plot with pandas
    theme_config = PlotTheme.apply_theme('professional')
    colors = get_palette(len(df))
    df.plot(kind='pie', subplots=True, figsize=(fig_width, fig_height), fontsize = annotation_textsize, autopct='%1.1f%%', colors=colors)
    
    plt.title(title, fontsize=title_fontsize)
    plt.ylabel("")
    plt.tight_layout()
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    

def vis_piechartWithExplode(df, groupby_col, explode, fig_width=8, fig_height=8, title='', title_fontsize =14,annotation_textsize = 12,
                 legend_title = '', legend_title_fontsize = 12, legend_fontsize = 12, fileName = ''):
    # Prepare Data
    df = df.groupby(groupby_col).size().reset_index(name='counts')

    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), subplot_kw=dict(aspect="equal"), dpi= 80)

    data = df['counts']
    categories = df[groupby_col]
    #explode = [0,0,0,0,0,0.1,0]

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}% ({:d} )".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(data, 
                                      autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"), 
                                      colors=get_palette(len(categories)),
                                      startangle=140,
                                      explode=explode
                                      )

    # Decoration
    ax.legend(wedges, categories, title = legend_title, loc = "center left", bbox_to_anchor = (1, 0, 0.5, 1), 
              fontsize = legend_fontsize, title_fontsize = legend_title_fontsize)
    plt.setp(autotexts, size=annotation_textsize, weight=700)
    ax.set_title(title, fontsize = title_fontsize)
    plt.tight_layout()
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
def vis_timeseries(df,date_col, fig_width = 14, fig_height = 8, text_fontsize = 14, title='', title_fontsize = 16, 
                   xticks_fontsize = 12, yticks_fontsize = 12, xlabel = '', ylabel = '', xlabel_fontsize = 10,
                   ylabel_fontsize = 10, xticks_interval = 6, fileName=''):
    """
    Module to plot all the numrical feature's trend present in the historical data.
    To get better overview, the dataset must be scaled with respect to each other.
    """

    # Draw Plot and Annotate
    theme_config = PlotTheme.apply_theme('professional')
    fig, ax = plt.subplots(1,1,figsize=(fig_width, fig_height), dpi= 80)    

    columns = df.columns[1:]
    colors = get_palette(len(columns))
    for i, column in enumerate(columns):    
        plt.plot(df[date_col].values, df[column].values, lw=1.5, color=colors[i])    
        plt.text(df.shape[0]+1, df[column].values[-1], column, fontsize=text_fontsize, color=colors[i])

    # Decorations    
    plt.tick_params(axis="both", which="both", bottom=False, top=False,    
                    labelbottom=True, left=False, right=False, labelleft=True)        

    # Lighten borders
    plt.gca().spines["top"].set_alpha(.3)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.3)
    plt.gca().spines["left"].set_alpha(.3)

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    
    plt.yticks(fontsize = xticks_fontsize)
    plt.xticks(range(0, df.shape[0], xticks_interval), df[date_col].values[::xticks_interval], horizontalalignment='left', fontsize=xticks_fontsize)   

    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
    
def vis_heatmap(df, fig_width = 12, fig_height=10, title='', title_fontsize=22, xticks_fontsize = 12, yticks_fontsize=12,  xlabel='',
                 ylabel='',xlabel_fontsize = 12, ylabel_fontsize = 12,fileName = ''):
    """
    This module helps us to analyze the relationship between all possible pairs of numeric variables in a given dataframe
    """
    
    # Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)

    # Decorations
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.tight_layout()
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
def vis_density_curve(df, category_col, value_col, fig_width = 12, fig_height = 6, title = '', title_fontsize = 16,
                          xlabel='', ylabel = '', xticks_fontsize = 12, yticks_fontsize = 12, legend_fontsize = 12 , xlabel_fontsize = 12,
                          ylabel_fontsize = 12, label='',cat_list=[], fileName = ''):
    """
    This module draws a Density curve with histogram which brings together the collective information conveyed by
    the two plots so that we can have them both in a single figure instead of two.
    """
    
    # Creating list of all unique categories in case category list is empty
    if len(cat_list) == 0:
        cat_list = list(df[category_col].unique())
    
    # Creating list of Random Colors
    colors = get_palette(len(cat_list))
    
    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    for i,category in enumerate(cat_list):
        sns.histplot(df.loc[df[category_col] == category, value_col], color=colors[i], label=category, kde=True, stat="density", alpha=.7)

    # Beautifying the plot visualisation
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(loc='upper left',fontsize = legend_fontsize)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
def vis_timelineWithPeaksAndThroughs(df, date_col, value_col, fig_width = 12, fig_height = 6, title = '', title_fontsize = 16,
                          xlabel='', ylabel = '', xticks_fontsize = 12, yticks_fontsize = 12, legend_fontsize = 12 ,
                                     xlabel_fonsize = 12, ylabel_fonsize = 12, label='', fileName = ''):
    """
    Module to plot the ups and downs of a feature in a given time.
    """
    # Get the Peaks and Troughs
    data = df[value_col].values
    doublediff = np.diff(np.sign(np.diff(data)))
    peak_locations = np.where(doublediff == -2)[0] + 1

    doublediff2 = np.diff(np.sign(np.diff(-1*data)))
    trough_locations = np.where(doublediff2 == -2)[0] + 1

    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    plt.plot(date_col, value_col, data=df, color='tab:blue', label=label)
    plt.scatter(df[date_col][peak_locations], df[value_col][peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
    plt.scatter(df[date_col][trough_locations], df[value_col][trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')

    # Annotate
    for t, p in zip(trough_locations[1::5], peak_locations[::3]):
        plt.text(df[date_col][p], df[value_col][p]+15, df[date_col][p], horizontalalignment='center', color='darkgreen')
        plt.text(df[date_col][t], df[value_col][t]-35, df[date_col][t], horizontalalignment='center', color='darkred')

    # Decoration
    xtick_location = df.index.tolist()[::6]
    xtick_labels = df[date_col].tolist()[::6]
    plt.xlabel(xlabel, fontsize = xlabel_fonsize)
    plt.ylabel(ylabel, fontsize = ylabel_fonsize)
    plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=xticks_fontsize, alpha=.7)
    plt.title(title, fontsize=title_fontsize)
    plt.yticks(fontsize=yticks_fontsize, alpha=.7)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.legend(loc='upper left',fontsize = legend_fontsize)
    plt.grid(axis='y', alpha=.3)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
def vis_timeline(df, date_col, value_col, fig_width = 12, fig_height = 6, title = '', title_fontsize = 16,
                 xlabel='', ylabel = '', xticks_fontsize = 12, yticks_fontsize = 12, legend_fontsize = 12 ,
                 xlabel_fonsize = 12, ylabel_fonsize = 12, label='', fileName = ''):
    """
    This module plots the journey of single given feature against the timeline
    """
    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    plt.plot(date_col, value_col, data=df, color='tab:blue', label=label)

    # Decoration
    xtick_location = df.index.tolist()[::6]
    xtick_labels = df[date_col].tolist()[::6]
    plt.xlabel(xlabel, fontsize = xlabel_fonsize)
    plt.ylabel(ylabel, fontsize = ylabel_fonsize)
    plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=xticks_fontsize, alpha=.7)
    plt.title(title, fontsize=title_fontsize)
    plt.yticks(fontsize=yticks_fontsize, alpha=.7)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)

    plt.legend(loc='upper left',fontsize = legend_fontsize)
    plt.grid(axis='both', alpha=.3)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    


def vis_bar(df,col, fig_width=12,fig_height=6,annotation_fontsize=10, title='',title_fontsize=16,ylabel='', ylabel_fontsize=12,xlabel='',
            xlabel_fontsize=12, xticks_fontsize=10, yticks_fontsize=10, xticks_rotation_angle = 60, fileName = ''):
    
    """
    Create a bar chart to visualise feature based on their count.
    """
    
    # Preparing Dataset
    df = df.groupby(col).size().reset_index(name='counts')
     
    # Using up different color codes to distinguish between the various classes/categories present
    n = df[col].unique().__len__()+1
    all_colors = list(plt.cm.colors.cnames.keys())
    random.seed(100)
    c = random.choices(all_colors, k=n)

    # Plot Bars
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    plt.bar(df[col], df['counts'], color=c, width=.5)
    for i, val in enumerate(df['counts'].values):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':annotation_fontsize})

    # Beautification
    plt.gca().set_xticklabels(df[col], rotation=xticks_rotation_angle, horizontalalignment= 'right')
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    


def vis_orderedBar(df,col, fig_width=12,fig_height=6,annotation_fontsize=10, title='',title_fontsize=16,ylabel='', ylabel_fontsize=12,xlabel='',
            xlabel_fontsize=12, xticks_fontsize=10, yticks_fontsize=10, xticks_rotation_angle = 60, fileName = ''):
    
    """
    Create an ordered bar chart to visualise feature based on their count.
    """
    
    # Preparing Dataset
    df = df.groupby(col).size().reset_index(name='counts').sort_values(['counts'], ascending=False)
     
    # Using up different color codes to distinguish between the various classes/categories present
    n = df[col].unique().__len__()+1
    all_colors = list(plt.cm.colors.cnames.keys())
    random.seed(100)
    c = random.choices(all_colors, k=n)

    # Plot Bars
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    plt.bar(df[col], df['counts'], color=c, width=.5)
    for i, val in enumerate(df['counts'].values):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':annotation_fontsize})

    # Beautification
    plt.gca().set_xticklabels(df[col], rotation=xticks_rotation_angle, horizontalalignment= 'right')
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
    
def vis_countPlot(df, col1, col2, fig_width = 12, fig_height = 6,title = '', title_fontsize = 14, xlabel = '', 
                  xlabel_fontsize = 12, ylabel = '', ylabel_fontsize = 12, xticks_fontsize = 10, yticks_fontsize=10, fileName = ''):
    
    """
    Module to visualize the correlation between two features using dots to depict the concentration.
    Larger the size of dot, more is the density accross that point.
    """
    
    # Preparing dataset
    df_counts = df.groupby([col1, col2]).size().reset_index(name='counts')

    # Draw Stripplot
    theme_config = PlotTheme.apply_theme('professional')
    fig, ax = plt.subplots(figsize=(fig_width,fig_height), dpi= 80)    
    sns.stripplot(data=df_counts, x=col2, y=col1, s=df_counts.counts*2, ax=ax)

    # Beautification
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    
    if fileName != '':
        plt.savefig(fileName)
    
    plt.show()
    


def vis_lollipop(df, groupby_col, mean_col,fig_width = 10, fig_height = 6, title='', title_fontsize = 14, xlabel = '',xlabel_fontsize = 10, 
                 ylabel = '',  ylabel_fontsize = 10,  xtick_fontsize = 10, xticks_rotation_angle = 60, ytick_fontsize = 10,
                 annotation_fontsize = 10, fileName = ''):
    """
    Lollipop is just another way of depicting the frequency of given feature
    """
    
    # Preparing dataset
    df = df[[mean_col, groupby_col]].groupby(groupby_col).apply(lambda x: x.mean())
    df.sort_values(mean_col, inplace=True)
    df.reset_index(inplace=True)
    
    # Draw plot
    theme_config = PlotTheme.apply_theme('professional')
    fig, ax = plt.subplots(figsize=(fig_width,fig_height), dpi= 80)
    ax.vlines(x=df.index, ymin=0, ymax=df[mean_col], color='firebrick', alpha=0.7, linewidth=2)
    ax.scatter(x=df.index, y=df[mean_col], s=75, color='firebrick', alpha=0.7)

    # Title, Label, Ticks and Ylim
    ax.set_title(title, fontdict={'size':title_fontsize})
    ax.set_xlabel(xlabel, fontsize= xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize= ylabel_fontsize)
    ax.set_xticks(df.index)
    ax.set_xticklabels(df[groupby_col].str.upper(), rotation=xticks_rotation_angle, fontdict={'horizontalalignment': 'right', 'size':xtick_fontsize})
    ax.tick_params(axis='y', labelsize=ytick_fontsize)
    ax.set_ylim(0, df[mean_col].max() + 20)
    
    # Annotate
    for row in df.itertuples():
        ax.text(row.Index, row.cty+.5, s=round(row.cty, 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=annotation_fontsize)

    if fileName != '':
        plt.savefig(fileName)

    plt.show()


def vis_populationPyramid(df, group_col, x_col, y_col, fig_width = 10, fig_height = 6, title = '', title_fontsize = 16,
                          xlabel='', ylabel = '', xticks_fontsize = 12, yticks_fontsize = 12, legend_fontsize = 12 , 
                          xlabel_fonsize = 12, ylabel_fonsize = 12, fileName = ''):
    
    
    
    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width, fig_height), dpi= 80)

    order_of_bars = df[y_col].unique()[::-1]
    colors = get_palette(len(df[group_col].unique()))

    for c, group in zip(colors, df[group_col].unique()):
        sns.barplot(x = x_col, y = y_col, data = df.loc[df[group_col]==group, :], order=order_of_bars, color=c, label=group)

    # Decorations    
    plt.xlabel(xlabel, fontsize = xlabel_fonsize)
    plt.ylabel(ylabel, fontsize = ylabel_fonsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.title(title, fontsize=title_fontsize)
    plt.legend(fontsize =legend_fontsize)
    plt.tight_layout()
    if fileName != '':
        plt.savefig(fileName)
    
    plt.show()
    
    
def vis_stackedAreaPlot(df,x_col,y_col_list, fig_width=12,fig_height=7, title='', title_fontsize=14, legend_fontsize =10, xtick_fontsize=10,
                        ytick_fontsize=10, xlabel='', ylabel = '',xlabel_fontsize = 12, ylabel_fontsize = 12, x_tick_interval = 50,fileName = '' ):
    
    
    # List Colors 
    colors = get_palette(len(df.columns))   

    # Draw Plot and Annotate
    theme_config = PlotTheme.apply_theme('professional')
    fig, ax = plt.subplots(1,1,figsize=(fig_width, fig_height), dpi= 80)
    columns = df[y_col_list]
    labs = y_col_list

    # Prepare data
    x  = df[x_col].values.tolist()
    
    temp_list = []
    for col in y_col_list:
        temp_list.append(df[col].values.tolist())
    y = np.vstack(temp_list)
    
    # Plot for each column
    ax = plt.gca()
    ax.stackplot(x, y, labels=labs, colors=colors, alpha=0.8)

    # Decorations
    ax.set_title(title, fontsize=title_fontsize)
    ax.legend(fontsize=legend_fontsize, ncol=4)
    plt.xticks(x[::x_tick_interval], fontsize=xtick_fontsize, horizontalalignment='center')
    plt.yticks(fontsize=ytick_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xlim(x[0], x[-1])

    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    
    if fileName != '':
        plt.savefig(fileName)
        
    plt.show()


def vis_unstackedAreaPlot(df, x_col, y_col_list, fig_width=12,fig_height=7, title='', title_fontsize=14, legend_fontsize =10, xtick_fontsize=10,
                        ytick_fontsize=10, xlabel='', ylabel = '',xlabel_fontsize = 12, ylabel_fontsize = 12, x_tick_interval = 5,fileName = '' ):
    
    
    # Decide Colors 
    colors = get_palette(len(y_col_list))      

    # Draw Plot and Annotate
    theme_config = PlotTheme.apply_theme('professional')
    columns = y_col_list#df[y_col_list]
    labs = y_col_list

    # Prepare data
    x  = df[x_col].values.tolist()
    
    temp_list = []
    for col in y_col_list:
        temp_list.append(df[col].values.tolist())
    
    #ax = plt.gca()
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi= 80)
    for i, y_col in enumerate(temp_list):
        ax.fill_between(x, y1=temp_list[i], y2=0, label=y_col_list[i], alpha=0.5, color=colors[i], linewidth=2)
        #ax.fill_between(x, y1=y_col, y2=0, alpha=0.3, linewidth = 2, label=y_col_list[i])


    # Decorations
    ax.set_title(title, fontsize=title_fontsize)
    ax.legend(fontsize=legend_fontsize, ncol=4)
    plt.xticks(x[::x_tick_interval], fontsize=xtick_fontsize, horizontalalignment='center')
    plt.yticks(fontsize=ytick_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xlim(x[0], x[-1])

    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    

def vis_scatter_with_hist(df,  x_col, y_col, right_hist_binsize, bottom_hist_binsize, fig_width = 14, fig_height = 8,
                          title = 'Scatterplot with Histograms', xLabel='X Label', yLabel = 'Y Label', xlabel_fontsize = 12, ylabel_fontsize = 12,
                              title_fontsize = 16, item_fontsize = 12, fileName = ''):
    
    # Create Fig and gridspec
    theme_config = PlotTheme.apply_theme('professional')
    fig = plt.figure(figsize=(fig_width, fig_height), dpi= 80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot on main ax
    ax_main.scatter(x_col, y_col, alpha=.9, data=df, cmap="tab10", edgecolors='gray', linewidths=.5)

    # histogram on the right
    ax_bottom.hist(df[x_col], right_hist_binsize, histtype='stepfilled', orientation='vertical', color='deeppink')
    ax_bottom.invert_yaxis()

    # histogram in the bottom
    ax_right.hist(df[y_col], bottom_hist_binsize, histtype='stepfilled', orientation='horizontal', color='deeppink')

    # Decorations
    ax_main.set(title=title, xlabel=xLabel, ylabel=yLabel)
    ax_main.title.set_fontsize(title_fontsize)
   
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(item_fontsize)
    ax_main.xaxis.label.set_fontsize(xlabel_fontsize)
    ax_main.yaxis.label.set_fontsize(ylabel_fontsize)

    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
    
def vis_scatter_with_box(df,  x_col, y_col, right_hist_binsize, bottom_hist_binsize, fig_width = 14, fig_height = 8,
                          title = 'Scatterplot with Histograms', xLabel='X Label', yLabel = 'Y Label',
                         xlabel_fontsize = 12, ylabel_fontsize = 12, title_fontsize = 16, item_fontsize = 12, fileName = ''):
    
    # Create Fig and gridspec
    theme_config = PlotTheme.apply_theme('professional')
    fig = plt.figure(figsize=(fig_width, fig_height), dpi= 80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Define the axes
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot on main ax
    ax_main.scatter(x_col, y_col, alpha=.9, data=df, cmap="tab10", edgecolors='gray', linewidths=.5)
    
    sns.boxplot(df[y_col], ax=ax_right, orient="v", color='deeppink')
    sns.boxplot(df[x_col], ax=ax_bottom, orient="h", color='deeppink')

    ax_bottom.set(xlabel='')
    ax_right.set(ylabel='')

    # Decorations
    ax_main.set(title=title, xlabel=xLabel, ylabel=yLabel)
    ax_main.title.set_fontsize(title_fontsize)
   
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(item_fontsize)
    ax_main.xaxis.label.set_fontsize(xlabel_fontsize)
    ax_main.yaxis.label.set_fontsize(ylabel_fontsize)

    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    

def vis_calendarHeatMap(df, date_col, year, feature, fig_width = 16, fig_height = 10, title = '', title_fontsize = 14,
                        label_fontsize = 14, fileName = ''):
    """
    Module to Plot a timeseries as a calendar heatmap.
    """
    df.set_index(date_col, inplace=True)

    df.head()

    # Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width, fig_height), dpi= 80)
    calmap.calendarplot(df[year][feature], fig_kws={'figsize': (fig_width, fig_height)}, yearlabel_kws={'color':'black', 'fontsize':label_fontsize}, subplot_kws={'title':title})
    plt.tight_layout()
    if fileName != '':
        plt.savefig(fileName)
    plt.show()


def vis_crossCorrelation(df, col1, col2, fig_width=12,fig_height=7, title = '', title_fontsize=14, xlabel = '', ylabel = '',
                       xlabel_fontsize = 12, ylabel_fontsize = 12, xticks_fontsize = 10, yticks_fontsize = 10, fileName = ''):
    x = df[col1]
    y = df[col1]

    # Computing Cross Correlations
    ccs = stattools.ccf(x, y)[:100]
    nlags = len(ccs)

    # Compute the Significance level
    conf_level = 2 / np.sqrt(nlags)

    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)

    plt.hlines(0, xmin=0, xmax=100, color='gray')  # 0 axis
    plt.hlines(conf_level, xmin=0, xmax=100, color='gray')
    plt.hlines(-conf_level, xmin=0, xmax=100, color='gray')

    plt.bar(x=np.arange(len(ccs)), height=ccs, width=.3)

    # Decoration
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xlim(0,len(ccs))
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    

def vis_categoricalPlot(df, feature1, feature2 ):
    """
    This module can be utilised to visualise the frequency distribution of 
    two or more categorical features in relation to each other
    """

    # Plot
    theme_config = PlotTheme.apply_theme('professional')
    g = sns.catplot(data=df[df[feature2].notnull()], x=feature1, col=feature2, col_wrap=4, kind="count", height=3.5, aspect=.8, palette='tab20')
    plt.show()
    
    
def vis_treemap(df, groupby_col, fig_width=10, fig_height=6, title='', title_fontsize =14,annotation_textsize = 12,
                 fileName = ''):
    """
    """
    # Prepare Data
    df = df.groupby(groupby_col).size().reset_index(name='counts')
    labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
    sizes = df['counts'].values.tolist()
    colors = get_palette(len(labels))

    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8, text_kwargs={'fontsize':annotation_textsize})

    # Decorate
    plt.title(title, fontsize = title_fontsize)
    plt.axis('off')
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
# Import Data
def vis_jitteringPlot(df, x_col, y_col, fig_width=12,fig_height=7, title = '', title_fontsize=14, xlabel = '', ylabel = '',
                       xlabel_fontsize = 12, ylabel_fontsize = 12, xticks_fontsize = 10, yticks_fontsize = 10, fileName = ''):

    # Draw Stripplot
    theme_config = PlotTheme.apply_theme('professional')
    fig, ax = plt.subplots(figsize=(fig_width,fig_height), dpi= 80)    
    sns.stripplot(data=df, x=x_col, y=y_col, jitter=0.25, s=8, ax=ax, linewidth=.5)

    # Decorations
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    

def vis_dotPlot(x, y, yticks_labels, fig_width=12, fig_height=7, title = '', title_fontsize=14, xlabel = '', ylabel = '',
                       xlabel_fontsize = 12, ylabel_fontsize = 12, xticks_fontsize = 10, yticks_fontsize = 10, fileName = ''):

    # Draw plot
    theme_config = PlotTheme.apply_theme('professional')
    fig, ax = plt.subplots(figsize=(fig_width,fig_height), dpi= 80)

    ax.scatter(y=y, x=x, s=75, color='firebrick', alpha=0.7)

    # Title, Label, Ticks and Ylim
    ax.set_title(title, fontdict={'size':title_fontsize})
    ax.set_xlabel(xlabel, fontdict={'size':xlabel_fontsize})
    ax.set_ylabel(ylabel, fontdict={'size':ylabel_fontsize})
    ax.set_yticks(y)
    ax.set_yticklabels(yticks_labels, fontdict={'horizontalalignment': 'right', 'size':yticks_fontsize})
    plt.xticks(fontsize=xticks_fontsize)
    ax.hlines(y=y, xmin=ax.get_xticks()[0], xmax=ax.get_xticks()[-1], color='gray', alpha=0.7, linewidth=1, linestyles='dashdot')
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    
    
def horizontalBargraph(x, y, fig_width=12, fig_height=7, title = '', title_fontsize=14, xlabel = '', ylabel = '',
                       xlabel_fontsize = 12, ylabel_fontsize = 12, xticks_fontsize = 10, yticks_fontsize = 10, fileName = '' ):
    
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(2, figsize=(fig_width, fig_height))
    #sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.0})
    sns.barplot(y = y, x = x, palette='husl', orient='h')
   
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    for y1, x1 in enumerate(x):
        plt.annotate(str(x1), xy=(x1, y1), va='center')
    plt.tight_layout()
    if fileName != '':
        plt.savefig(fileName)
    plt.show()

    
def vis_violen(df, x_col, y_col,  fig_width = 7, fig_height = 5, title = '', title_fontsize =14,xlabel = '', ylabel = '',
                       xlabel_fontsize = 12, ylabel_fontsize = 12, xticks_fontsize = 10, yticks_fontsize = 10, fileName = ''):
    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)
    sns.violinplot(x=x_col, y=y_col, data=df, scale='width', inner='quartile')

    # Decoration
    plt.title(title, fontsize=title_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
    

def vis_densityPlot(df, category_col, x_col, cat_list, fig_width=10, fig_height=6, title = '', legend_fontsize =10, title_fontsize=14, xlabel = '', ylabel = '',xlabel_fontsize = 12, ylabel_fontsize = 12, xticks_fontsize = 10, yticks_fontsize = 10, fileName = ''):
    
    # Draw Plot
    theme_config = PlotTheme.apply_theme('professional')
    plt.figure(figsize=(fig_width,fig_height), dpi= 80)

    if len(cat_list) == 0:
        cat_list = df[cat_col].unique()
        
    colors = get_palette(len(cat_list)) 
        
    for i, category in enumerate(cat_list):
        sns.kdeplot(df.loc[df[category_col] == category, x_col], fill=True, color=colors[i], label=str(category_col)+":"+str(category), alpha=.7)

    # Decoration
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    plt.xticks(fontsize=xticks_fontsize)
    plt.yticks(fontsize=yticks_fontsize)
    plt.legend(fontsize =legend_fontsize)
    if fileName != '':
        plt.savefig(fileName)
    plt.show()
