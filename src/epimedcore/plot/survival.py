from src.epimedcore.service import FigureService, SurvivalService
from src.epimedcore.plot import Plot
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

# ==============================

class KaplanMeierPlot(Plot):
    
    def __init__(self, 
             ax, 
             continuous_series: pd.Series,
             categorical_series: pd.Series,
             survival_data: pd.DataFrame,
             survival_type: str = 'os',
             max_survival_time: float = 120,
             **kwargs
             ):
        
        self.ax = ax
        self.continuous_series = continuous_series
        self.categorical_series = categorical_series
        self.survival_data = survival_data
        self.survival_type = survival_type
        self.max_survival_time = max_survival_time
        self.duration_col = survival_type + '_months'
        self.event_col = survival_type + '_censor'
        self.group_names = None
        self.group_colors = None
        self.regular = 20
        self.title = ''
        self.legend_position = 'lower left'
        self.show_title = True
        self.show_labels = True
        self.show_pvalues = True
        self.show_legend = True
        self.step_time = 20
        self.show_cindex = False
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self._shifted_survival_data = None
        self._kmf = KaplanMeierFitter()
        self._cph = CoxPHFitter()
        
        self.logrank_pvalue = np.nan
        self.cox_pvalue = np.nan
        self.cox_hr = np.nan
        self.cox_bin_pvalue = np.nan
        self.cox_bin_hr = np.nan
        self.c_index = np.nan
        
        self.elements = dict()
        self.encoded_group_names = dict()
        self.encoded_group_colors = dict()
        
    def plot(self) -> None:
        self.encode_data()
        self.generate_shifted_survival_data()
        self.calculate_logrank()
        self.calculate_cox_model()
        groups = sorted(list(self._shifted_survival_data['feature'].unique()))
        self.generate_group_colors(groups)
        
        arial_narrow_font = FigureService.create_arial_narrow_font()
        regular, medium, small, tiny = FigureService.create_font_sizes(regular=self.regular)
        
        title = self.title
        if self.show_pvalues:
            logrank_pvalue_text = 'logrank p-value = ' + '{:.1e}'.format(self.logrank_pvalue) + ' ' + FigureService.get_significance_symbol(self.logrank_pvalue)
            logrank_pvalue_text = logrank_pvalue_text.strip()
            cox_pvalue_text = 'cox p-value = ' + '{:.1e}'.format(self.cox_pvalue) + ' ' + FigureService.get_significance_symbol(self.cox_pvalue)
            cox_pvalue_text = cox_pvalue_text.strip()
            cox_bin_hr_text = 'hazard ratio = ' + '{:.1f}'.format(self.cox_bin_hr)
            title = title + '\n' + cox_pvalue_text + '\n' + logrank_pvalue_text + '\n' + cox_bin_hr_text
            if self.show_cindex:
                c_index_text = 'c-index = ' + '{:.3f}'.format(self.c_index)
                title = title + '\n' + c_index_text
        
        if self.show_title:
            self.ax.set_title(title, fontsize=regular, **arial_narrow_font)       
        
        self.ax.set_ylim([-0.05, 1.05])
        self.ax.set_xlim([0.0 - 0.05*self.max_survival_time, self.max_survival_time + 0.05*self.max_survival_time])
        
        xticks = np.arange(0, self.max_survival_time + self.step_time, self.step_time)
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xticks, **arial_narrow_font)
        
        yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.ax.set_yticks(yticks)
        self.ax.set_yticklabels(yticks, **arial_narrow_font)
        
        for group in groups:
            group_data = self._shifted_survival_data[self._shifted_survival_data['feature']==group]
            group_data = group_data.dropna(axis=0)
            group_label = f"{self.encoded_group_names[group]} (n={group_data.shape[0]})"
            self._kmf.fit(group_data['time'], group_data['event'], label=group_label)
            self._kmf.plot(ax=self.ax, ci_show=False, show_censors=False, color=self.encoded_group_colors[group], linewidth=2)
            # print(group, 'median survival', self._kmf.median_survival_time_)
            # print(group, 'survival function', self._kmf.survival_function_)
        
        L = self.ax.legend(fontsize=small, loc=self.legend_position)
        plt.setp(L.texts, **arial_narrow_font)
        
        if not self.show_legend:
            self.ax.get_legend().remove()
        
        
        if self.show_labels:
            survival_name = SurvivalService.get_survival_name(self.survival_type)
            self.ax.set_ylabel(survival_name, fontsize=regular, **arial_narrow_font)
            self.ax.set_xlabel('Time in months', fontsize=regular, **arial_narrow_font)
            self.ax.tick_params(axis='both', labelsize=medium)
        else:
            self.ax.set_xticklabels([])
            self.ax.set_xlabel('')
            self.ax.set_yticklabels([])
            self.ax.set_ylabel('')
        
        self.elements['legend'] = L
        self.elements['xticks'] = xticks
        self.elements['yticks'] = yticks
        
    
    def encode_data(self):
        encoder = LabelEncoder()
        self.categorical_series = self.categorical_series.dropna()
        encoded_data = encoder.fit_transform(self.categorical_series)
        self.categorical_series = pd.Series(encoded_data, index=self.categorical_series.index, dtype=int)
        
        if self.group_names is None:
            unique_encoded_values = sorted(list(self.categorical_series.unique()))
            categories = list(encoder.inverse_transform(unique_encoded_values))
            self.group_names = dict(zip(unique_encoded_values, categories))
   
        unique_values = list(self.group_names.keys())
        unique_encoded_values = encoder.fit_transform(unique_values)
        
        if self.group_colors is None:
            self.generate_group_colors(unique_encoded_values)
   
        for k, ke in zip(unique_values, unique_encoded_values):
            self.encoded_group_names[ke] = self.group_names[k]
            self.encoded_group_colors[ke] = self.group_colors[k]
                 
        self.continuous_series = self.continuous_series.dropna()
        is_numeric = self.continuous_series.dtype=='int64' or self.continuous_series.dtype== 'float64'
        if not is_numeric:
            encoded_data = encoder.fit_transform(self.continuous_series)
            self.continuous_series = pd.Series(encoded_data, index=self.continuous_series.index, dtype=int)
    
    def generate_shifted_survival_data(self):
        self._shifted_survival_data = pd.DataFrame(index=self.categorical_series.index)
        for id_sample in self.categorical_series.index:
            shifted_time = self.survival_data.loc[id_sample, self.duration_col]
            shifted_event = self.survival_data.loc[id_sample, self.event_col]
            if (shifted_time > self.max_survival_time):
                shifted_time = self.max_survival_time
                shifted_event = 0.0
            self._shifted_survival_data.loc[id_sample, 'time'] = shifted_time
            self._shifted_survival_data.loc[id_sample, 'event'] = shifted_event
        self._shifted_survival_data['feature'] = self.categorical_series
        self._shifted_survival_data.dropna(axis=0)    
    
    def get_group_name(self, group) -> str:
        if (self.group_names is not None) and (group in self.group_names.keys()):
            return self.group_names[group]
        return str(group)
    
    def generate_group_colors(self, groups) -> None:
        if self.group_colors is None:
            group_colors = FigureService.extract_colors_from_colormap(n=len(groups), colormap='jet')
            self.group_colors = dict(zip(groups, group_colors))
    
    def calculate_cox_model(self):
        logrank_data = self._shifted_survival_data # binarized
        logrank_data = logrank_data.dropna(axis=0)
        common_samples = sorted(list(set(self.continuous_series.index).intersection(set(logrank_data.index))))
        cox_data = pd.DataFrame(index=common_samples)
        cox_data['time'] = logrank_data.loc[common_samples, 'time']
        cox_data['event'] = logrank_data.loc[common_samples, 'event']
        cox_data['feature'] = self.continuous_series.loc[common_samples]
        cox_data = cox_data.dropna(axis=0)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self._cph.fit(logrank_data, duration_col='time', event_col='event', show_progress=False)
                self.cox_bin_pvalue = self._cph.summary.p['feature']
                self.cox_bin_hr = self._cph.summary['exp(coef)']['feature']
                self.c_index = self._cph.concordance_index_
                self._cph.fit(cox_data, duration_col='time', event_col='event', show_progress=False)
                self.cox_pvalue = self._cph.summary.p['feature']
                self.cox_hr = self._cph.summary['exp(coef)']['feature']
                
        except:
            self.cox_pvalue = np.nan
            self.cox_hr = np.nan
            self.cox_bin_pvalue = np.nan
            self.cox_bin_hr = np.nan
            self.c_index = np.nan
        
            
    def calculate_logrank(self):
        logrank_data = self._shifted_survival_data
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                logrank = multivariate_logrank_test(logrank_data['time'], logrank_data['feature'], logrank_data['event'])
                self.logrank_pvalue = logrank.p_value
        except:
            self.logrank_pvalue = np.nan
