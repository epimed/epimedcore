from src.epimedcore.plot import Plot
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.cluster import hierarchy
import pingouin as pg
from src.epimedcore.service import FigureService
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

class AnovaBoxplot(Plot):
    
    def __init__(self,
                 ax, 
                 continuous_data: dict[str, pd.DataFrame],
                 feature,
                 group_names: list[str] = None,
                 ref_group_name: str = None,
                 **kwargs
                 ):
        
        self.ax = ax
        self.continuous_data = continuous_data
        self.feature = feature
        self.group_names = group_names
        self.ref_group_name = ref_group_name
        
        self.show_title = True
        self.show_labels = True
        self.show_pvalues = True
        self.group_colors = None
        self.regular = 20 
        self.title = ''
        self.boxplot_options = FigureService.create_boxplot_options()
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.anova_data = pd.DataFrame()
        self.anova_pvalue = np.nan
        self.boxplot_data = []
        self.ttest_pvalues = dict()
        
    def plot(self):
        self._set_group_names()
        self._set_ref_group_name()
        self._set_group_colors()
        self._generate_data()
        self._perform_anova()
        self._perform_ttest()
        self._add_boxplots()
        self._add_annotations()
    
    def _add_boxplots(self):
        text_colors = []
        bplot = self.ax.boxplot(self.boxplot_data, **self.boxplot_options)
        for patch, group_name in zip(bplot['boxes'], self.group_names):
            patch.set_facecolor(self.group_colors[group_name])
            patch.set_alpha(0.5)
            text_colors.append(self.group_colors[group_name])
        [t.set_color(text_colors[j]) for j, t in enumerate(self.ax.get_xticklabels())]    
        
    def _add_annotations(self):
        arial_narrow_font = FigureService.create_arial_narrow_font()
        regular, medium, small, tiny = FigureService.create_font_sizes(regular=self.regular)
        
        title = self.title
        if self.show_pvalues:
            pAnovaText = 'ANOVA p-value = ' + '{:.1e}'.format(self.anova_pvalue) + ' ' + FigureService.get_significance_symbol(self.anova_pvalue)
            pAnovaText = pAnovaText.strip()
            title = title + '\n' + pAnovaText
        
        if self.show_title:
            self.ax.set_title(title, fontsize=regular, **arial_narrow_font)
            
        xticks = [i+1 for i in range(len(self.group_names))]
        self.ax.set_xticks(xticks)
        self.ax.tick_params(axis='y', labelsize=tiny)
        
        if self.show_labels:
            xticklabels = []
            for group_name in self.group_names:
                ttest_pvalue_test = 'REF'
                if group_name in self.ttest_pvalues.keys():
                    ttest_pvalue_test = 'p = ' + '{:.1e}'.format(self.ttest_pvalues[group_name]) + ' ' + FigureService.get_significance_symbol(self.ttest_pvalues[group_name])
                    ttest_pvalue_test = ttest_pvalue_test.strip()
                xticklabel = group_name + '\n' + ' (n=' + str(self.continuous_data[group_name].shape[0])  + ')' + '\n' + ttest_pvalue_test
                xticklabels.append(xticklabel)
            self.ax.set_xticklabels(xticklabels, ha='center', fontsize=regular, **arial_narrow_font)
            self.ax.set_ylabel('Expression', fontsize=regular, **arial_narrow_font)
            
    
    
    def _perform_ttest(self):
        ref_group_data = list(self.continuous_data[self.ref_group_name][self.feature])
        for group_name in self.group_names:
            if group_name!= self.ref_group_name:
                group_data = list(self.continuous_data[group_name][self.feature])
                statistics, t_pvalue = ttest_ind(ref_group_data, group_data, nan_policy='omit')
                self.ttest_pvalues[group_name] = t_pvalue
                                 
        
    def _perform_anova(self):
        aov = pg.anova(data=self.anova_data, dv='feature', between='group_name', detailed=True)
        self.anova_pvalue = aov.loc[0, 'p-unc']
        
    def _generate_data(self):
        for group_name in self.group_names:
            for id_sample in self.continuous_data[group_name].index:
                self.anova_data.loc[id_sample, 'feature'] = self.continuous_data[group_name].loc[id_sample, self.feature]
                self.anova_data.loc[id_sample, 'group_name'] = group_name
            self.boxplot_data.append(list(self.continuous_data[group_name][self.feature]))
         
    def _set_group_names(self):
        if self.group_names is None:
            self.group_names = list(self.continuous_data.keys())
        
    def _set_ref_group_name(self):
        if self.ref_group_name is None:
            self.ref_group_name = self.group_names[0]
    
    def _set_group_colors(self) -> None:
        if self.group_colors is  None:
            group_colors = FigureService.extract_colors_from_colormap(n=len(self.group_names), colormap=Plot.default_colormap)
            self.group_colors = dict(zip(self.group_names, group_colors))
            

# ==============================  
            
class FrequencyBarplot(Plot):
    
    def __init__(self,
                 ax, 
                 frequency: pd.DataFrame,
                 feature,
                 group_names: list[str] = None,
                 **kwargs
                 ):
        
        self.ax = ax
        self.frequency = frequency
        self.feature = feature
        self.group_names = group_names

        self.show_title = True
        self.title = None
        self.group_colors = None
        self.sample_size = None
        self.regular = 20 
        self.barwidth = 0.7
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.elements = dict()
        

    def plot(self):
        self._set_group_names()
        self._set_group_colors()
        self._add_barplots()
    
    
    def _add_barplots(self):
        arial_narrow_font = FigureService.create_arial_narrow_font()
        regular, medium, small, tiny = FigureService.create_font_sizes(regular=self.regular)
        xticks = []
        xticklabels = []
        xtickcolors = []
        for i, group_name in enumerate(self.group_names):
            frequency_value = self.frequency.loc[self.feature, group_name]
            self.ax.bar(i, frequency_value, width=self.barwidth, align='center', color=self.group_colors[group_name], alpha=0.5)
            self.ax.text(i, frequency_value + 3.0, '{:.1f}'.format(frequency_value) + '%', ha='center', va='center', fontsize=regular, **arial_narrow_font, color=self.group_colors[group_name])
            xticks.append(i)
            xticklabels.append(self._generate_xticklabel(group_name))
            xtickcolors.append(self.group_colors[group_name])
        self.ax.set_xlim([min(xticks)-0.5, max(xticks)+0.5])
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xticklabels, rotation=0, fontsize=regular, **arial_narrow_font) 
        
        
        [t.set_color(xtickcolors[j]) for j, t in enumerate(self.ax.get_xticklabels())]
        
        yticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(yticks, **arial_narrow_font)
        self.ax.set_ylabel('Frequency', fontsize=regular, **arial_narrow_font)
        self.ax.set_ylim([0.0, 110.0])
        self.ax.tick_params(axis='y', labelsize=tiny)
        
        if self.show_title:
            title = self._generate_title()
            self.ax.set_title(title, fontsize=regular, **arial_narrow_font)
        
        self.elements['xticks'] = xticks
        self.elements['xticklabels'] = xticklabels
        self.elements['width'] = self.barwidth

        
    def _generate_title(self):
        if self.title is None:
            return self.feature
        return self.title
            
    def _generate_xticklabel(self, group_name: str): 
        if 'sample_size' in self.__dict__.keys():
            if self.sample_size is not None:
                if group_name in self.sample_size.keys():
                    return f"{group_name}\n(n={self.sample_size[group_name]})"
        return group_name  
    
    def _set_group_names(self):
        if self.group_names is None:
            self.group_names = list(self.frequency.columns)
    
    def _set_group_colors(self) -> None:
        if self.group_colors is  None:
            group_colors = FigureService.extract_colors_from_colormap(n=len(self.group_names), colormap=Plot.default_colormap)
            self.group_colors = dict(zip(self.group_names, group_colors))
            

# ==============================  
            
class FrequencyHeatmap(Plot):
    
    def __init__(self,
                 ax, 
                 frequency: pd.DataFrame,
                 features: list = None,
                 group_names: list[str] = None,
                 **kwargs
                 ):
        
        self.ax = ax
        self.frequency = frequency
        self.features = features
        self.group_names = group_names

        self.title = None
        self.regular = 20
        self.cmap = 'Greens'
        self.text_colors = {0.0: 'black', 50.0: 'white'}
        self.square = True
        self.order_by_group_name = None
        self.ascending = False
        self.vmin = 0.0
        self.center = 50.0
        self.vmax = 100.0
        self.text_rotation = True
        
        
        self.arial_narrow_font = FigureService.create_arial_narrow_font()
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        

    def plot(self):
        self._set_features()
        self._set_group_names()
        self._add_heatmap()
    
    
    def _add_heatmap(self):
        regular, medium, small, tiny = FigureService.create_font_sizes(regular=self.regular)
        frequency = self.frequency.loc[self.features, self.group_names]
        if self.order_by_group_name is not None:
            frequency = frequency.sort_values(by=self.order_by_group_name, ascending=self.ascending)
        X = frequency.T
        sns.heatmap(X, ax=self.ax, linewidths=2, linecolor='white', square=self.square, cmap=self.cmap, cbar=False, center=self.center, vmin=self.vmin, vmax=self.vmax)
        self.ax.set_yticklabels(X.index, ha='right', va='center', rotation=0, fontsize=regular, **self.arial_narrow_font)
        
        self.ax.xaxis.tick_top()
        self.ax.xaxis.set_label_position('top') 
        self.ax.set_xlabel('')
        self.ax.tick_params(which='major', bottom=False, left=False, top=False, right=False)
        
        if self.text_rotation:
            self.ax.set_xticklabels(X.columns, rotation=90, ha='left', va='center', rotation_mode='anchor', fontsize=regular, **self.arial_narrow_font)
        else:
            self.ax.set_xticklabels(X.columns, rotation=0, ha='center', va='center', fontsize=regular, **self.arial_narrow_font)
        
        
        self._add_text(X, tiny)
        
        if self.title is not None:
            self.ax.set_title(self.title, fontsize=regular, **self.arial_narrow_font)
            
     
    def _add_text(self, X: pd.DataFrame, fontsize: float):    
        for ind in np.arange(X.shape[0]):
            for col in np.arange(X.shape[1]):
                frequency = X.iloc[ind, col]
                if not np.isnan(frequency):
                    min_key = np.min(list(self.text_colors.keys()))
                    textcolor = self.text_colors[min_key]
                    for minval, color in self.text_colors.items():
                        if (frequency > minval):
                            textcolor = color
                    self.ax.text(col+0.5, ind+0.5, '{:.1f}'.format(frequency) + '%', ha='center', va='center', color=textcolor, fontsize=fontsize, **self.arial_narrow_font)
    
    
    def _set_features(self):
        if self.features is None:
            self.features = list(self.frequency.index)
    
    def _set_group_names(self):
        if self.group_names is None:
            self.group_names = list(self.frequency.columns)

        
# ==============================  

class FrequencyScatterplot(Plot):
    
    def __init__(self,
                 ax, 
                 continuous_data: dict[str, pd.DataFrame],
                 threshold_values: pd.Series,
                 feature,
                 group_names: list[str] = None,
                 ref_group_name = 'Normal',
                 **kwargs
                 ):
        
        self.ax = ax
        self.continuous_data = continuous_data
        self.threshold_values = threshold_values
        self.feature = feature
        self.group_names = group_names
        self.ref_group_name = ref_group_name
        
        self.show_title = True
        self.title = None
        self.group_colors = None
        self.sample_size = None
        self.regular = 20 
        self.show_threshold = True
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        
    def plot(self):
        self._set_group_names()
        self._set_group_colors()
        self._add_scatterplot()
        
    
    def _add_scatterplot(self):
        arial_narrow_font = FigureService.create_arial_narrow_font()
        regular, medium, small, tiny = FigureService.create_font_sizes(regular=self.regular) 
        xticks = []
        xticklabels = [] 
        for i, group_name in enumerate(self.group_names):
            data = self.continuous_data[group_name]
            xticks.append(i)
            xticklabels.append(self._generate_xticklabel(group_name))
            if group_name==self.ref_group_name:
                y = data[self.feature]
                x = i * np.ones(len(y))
                self.ax.scatter(x, y, s=50, facecolor=self.group_colors[group_name], edgecolor='whitesmoke', alpha=0.8, lw=0.5)
            else:
                filters = {
                    0: data[self.feature]<=self.threshold_values[self.feature],
                    1: data[self.feature]>self.threshold_values[self.feature]
                    }
                for subgroup, f in filters.items():
                    y = data[f][self.feature]
                    x = i * np.ones(len(y))
                    self.ax.scatter(x, y, s=50, facecolor=self.group_colors[subgroup], edgecolor='whitesmoke', alpha=0.8, lw=0.5)
                    frequency = 100.0 * y.shape[0]/data.shape[0]
                    self.ax.text(i+0.1, np.mean(y), '{:.1f}'.format(frequency) + '%', ha='left', va='center', color=self.group_colors[subgroup], fontsize=medium, **arial_narrow_font) 
        xmin = -0.5
        xmax = len(self.group_names)-0.5
        if self.show_threshold:
            self.ax.plot([xmin, xmax], [self.threshold_values[self.feature], self.threshold_values[self.feature]], linestyle='--', color='grey', alpha=0.5)
        self.ax.set_xlim([xmin, xmax])
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xticklabels, rotation=0, fontsize=regular, **arial_narrow_font)
        self.ax.set_ylabel('Expression', fontsize=regular, **arial_narrow_font)
        self.ax.tick_params(axis='y', labelsize=tiny)
        if self.show_title:
            title = self._generate_title()
            self.ax.set_title(title, fontsize=regular, **arial_narrow_font) 
        
    
    def _generate_title(self):
        if self.title is None:
            return self.feature
        return self.title
            
    def _generate_xticklabel(self, group_name: str): 
        if 'sample_size' in self.__dict__.keys():
            if self.sample_size is not None:
                if group_name in self.sample_size.keys():
                    return f"{group_name} \n(n={self.sample_size[group_name]})"
        return group_name  
        
    def _set_group_names(self):
        if self.group_names is None:
            self.group_names = list(self.continuous_data.keys())
    
    def _set_group_colors(self) -> None:
        if self.group_colors is None:
            group_colors = FigureService.extract_colors_from_colormap(n=len(self.group_names), colormap=Plot.default_colormap)
            self.group_colors = dict(zip(self.group_names, group_colors))
            self.group_colors[0] = 'royalblue'
            self.group_colors[1] = 'crimson'

# ==============================  

class Clustering(Plot):
    
    def __init__(self,
                 fig, 
                 continuous_data: dict[str, pd.DataFrame],
                 group_name: str = None,
                 features: list = None,
                 col_colors: pd.Series = None,
                 **kwargs
                 ):
        
        self.fig = fig
        self.continuous_data = continuous_data
        self.features = features
        self.group_name = group_name
        self.col_colors = col_colors
        
        self.metric = 'euclidean'
        self.method = 'ward'
        self.cmap = 'coolwarm'
        self.vlim = 10.0
        self.show_title = True
        self.title = None
        self.regular = 20
        self.link_color_palette = None # ['aquamarine', 'darkslateblue', 'darkorange'] list or cmap
        self.n_clusters = 3
        self.show_clusters = True
        self.cluster_colors = None # dict[int: color]
        self.cluster_data = None # pd.DataFrame(columns=['cluster', 'color'])

        for k, v in kwargs.items():
            setattr(self, k, v)


    def plot(self):
        self._set_group_name()
        self._set_features()
        
        arial_narrow_font = FigureService.create_arial_narrow_font()
        regular, medium, small, tiny = FigureService.create_font_sizes(regular=self.regular) 
        
        data = self.continuous_data[self.group_name][self.features]
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(data)
        X_scaled = pd.DataFrame(X_scaled, index=data.index, columns=data.columns)
        
        vmin = X_scaled.min().min()
        vmax = X_scaled.max().max()
        vmax = max(abs(vmin), abs(vmax))
        vmax = round(min(vmax, self.vlim), 1)
        vmin = -vmax
        
        margin = 0.05
        x_hm = 0.2
        y_hm = 0.25
        h_hm = 0.5
        
        w_den_row = 0.15
        h_den_col = 0.15
        y_den_col = 1.0 - margin - h_den_col
        x_den_row = x_hm - 0.015 - w_den_row
        
        h_cbar = 0.05
        y_cbar = 0.1
        
        h_cmap = 0.03
        y_cmap = y_hm + h_hm + 0.01
        
        ax_hm = self.fig.add_axes([x_hm, y_hm, 0.7, h_hm])
        ax_den_row = self.fig.add_axes([x_den_row, y_hm, w_den_row, h_hm])
        ax_den_col = self.fig.add_axes([x_hm, y_den_col, 0.7, h_den_col])
        ax_cbar = self.fig.add_axes([x_hm, y_cbar, 0.7, h_cbar])
        ax_cmap = self.fig.add_axes([x_hm, y_cmap, 0.7, h_cmap])
        
        hierarchy.set_link_color_palette(['grey'])  
        if self.link_color_palette is not None:
            hierarchy.set_link_color_palette(self.link_color_palette)
        Z_col = hierarchy.linkage(X_scaled, method=self.method, metric=self.metric)
        clusters = hierarchy.fcluster(Z_col, t=self.n_clusters, criterion='maxclust')
        Z_row = hierarchy.linkage(X_scaled.T, method=self.method, metric=self.metric)
        
        with plt.rc_context({'lines.linewidth': 1.0}):
            dn_col = hierarchy.dendrogram(Z_col, ax=ax_den_col, above_threshold_color='grey', orientation='top')
            dn_row = hierarchy.dendrogram(Z_row, ax=ax_den_row, above_threshold_color='grey', orientation='left')
            
        ind_col = dn_col['leaves']
        ind_row = dn_row['leaves']
        
        heatmap = X_scaled.T.iloc[ind_row, ind_col]
        cbar_kws = {'use_gridspec': False, 'orientation': 'horizontal'}
        ax_hm = sns.heatmap(heatmap, ax=ax_hm, linewidths=0.0, cmap=self.cmap, cbar=True, cbar_ax=ax_cbar, cbar_kws=cbar_kws, center=0.0, vmin=vmin, vmax=vmax)
        
        ax_hm.set_xlabel('')
        ax_hm.set_xticks([])
        ax_hm.set_xticklabels([])
        
        ax_hm.set_ylabel('')
        ax_hm.yaxis.set_label_position('right')
        ax_hm.yaxis.tick_right()
        
        ax_hm.set_yticklabels(heatmap.index, ha='left', va='center', rotation=0, fontsize=tiny, **arial_narrow_font)
        
        self.cluster_data = pd.DataFrame(index=X_scaled.index)
        self.cluster_data['cluster'] = clusters
        
        
        if self.show_clusters:
            self._generate_col_colors()
            y = self.col_colors.loc[heatmap.columns] 
            ax_cmap.set_xlim([0, len(ind_col)])
            ax_cmap.set_ylim([0, 1])
            for i, yc in enumerate(y):
                ax_cmap.fill_between([i, i+1], [1, 1], color=yc)
        
        ax_cbar.set_title('Normalized expression level', fontsize=medium, **arial_narrow_font)
    
        if self.show_title:    
            title = self._generate_title()
            ax_den_col.set_title(title, fontsize=regular, **arial_narrow_font)

        ax_den_row.set_axis_off()
        ax_den_col.set_axis_off()
        ax_cmap.set_axis_off()
    
    
    def _generate_col_colors(self):
        if self.col_colors is None:
            n_clusters = sorted(list(self.cluster_data['cluster'].unique()))
            if self.cluster_colors is None: 
                default_colors = FigureService.extract_colors_from_colormap(len(n_clusters), Plot.default_colormap)
                self.cluster_colors = dict(zip(n_clusters, default_colors))
            self.col_colors = self.cluster_data['cluster'].map(self.cluster_colors)
            self.cluster_data['color'] = self.col_colors
            
    def _generate_title(self):
        if self.title is None:
            return self.group_name
        return self.title
    
    def _set_group_name(self):
        if self.group_name is None:
            self.group_name = list(self.continuous_data.keys())[0]        
            
    def _set_features(self):
        if self.features is None:
            self.features = list(self.continuous_data[self.group_name].columns)
            
            
# ==============================  

class TotalCombinationsBarplot(Plot):
    
    def __init__(self,
                 ax, 
                 top_combinations: dict[str, pd.DataFrame],
                 group_name: str = None,
                 **kwargs
                 ):
        
        self.ax = ax
        self.top_combinations = top_combinations
        self.group_name = group_name

        self.min_subset_size = None
        self.max_subset_size = None
        self.n_features_colname = 'n_features'
        self.frequency_colname = 'frequency'
        self.combination_colname = 'combination'

        self.width = 0.7
        self.show_title = True
        self.title = None
        self.group_colors = None
        self.sample_size = None
        self.regular = 20
        self.bar_color = 'royalblue'
        self.text_color = 'darkblue'
        self.ytext = -30.0
        self.ystep = 8.0
        self.xlabel = 'Number of genes in subset'
        self.ylabel = 'Frequency'
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        
    def plot(self):
        self._set_group_name()
        self._set_subset_size()
        self._add_boxplot()
    
    def _add_boxplot(self):
        arial_narrow_font = FigureService.create_arial_narrow_font()
        regular, medium, small, tiny = FigureService.create_font_sizes(regular=self.regular)
        top_combinations = self.top_combinations[self.group_name]
        range_n_features = np.arange(self.min_subset_size, self.max_subset_size+1, 1)
        ytext = self.ytext
        ystep = self.ystep 
        for n in range_n_features:
            filter_n = top_combinations[self.n_features_colname]==n
            n_top = top_combinations[filter_n].sort_values(by=self.frequency_colname, ascending=False)
            n_top_frequency = n_top.iloc[0][self.frequency_colname]
            n_top_combination = n_top.iloc[0][self.combination_colname]
            self.ax.bar(n, n_top_frequency, self.width, align='center', color=self.bar_color, alpha=0.5)
            n_top_frequency_text = '{:.1f}'.format(n_top_frequency) + '%'
            self.ax.text(n, n_top_frequency + 3.0, n_top_frequency_text, ha='center', va='center', fontsize=medium, **arial_narrow_font, color=self.text_color)
            text_details = '{:.0f}'.format(n) + '   ' + '{:.1f}'.format(n_top_frequency) + '%' + '   ' + n_top_combination
            self.ax.text(self.min_subset_size-0.5, ytext, text_details, ha='left', va='center', fontsize=small, **arial_narrow_font)
            ytext = ytext - ystep
        
        yticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(yticks, **arial_narrow_font)
        self.ax.set_ylabel(self.ylabel, fontsize=regular, **arial_narrow_font)
        self.ax.set_ylim([0.0, 110.0])
        self.ax.tick_params(axis='y', labelsize=tiny)
        
        xticks = [int(n) for n in range_n_features] 
        self.ax.set_xlim([min(xticks)-0.5, max(xticks)+0.5])
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xticks, rotation=0, fontsize=medium, **arial_narrow_font)
        self.ax.set_xlabel(self.xlabel, fontsize=regular, **arial_narrow_font)  
        
        if self.show_title:    
            title = self._generate_title()
            self.ax.set_title(title, fontsize=regular, **arial_narrow_font)         


    def _generate_title(self):
        if self.title is None:
            return self.group_name
        return self.title

    def _set_subset_size(self):
        if self.min_subset_size is None:
            self.min_subset_size = self.top_combinations[self.group_name][self.n_features_colname].min()
        if self.max_subset_size is None:
            self.max_subset_size = self.top_combinations[self.group_name][self.n_features_colname].max()
        
    def _set_group_name(self):
        if self.group_name is None:
            self.group_name = list(self.top_combinations.keys())[0]    