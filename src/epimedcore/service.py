import os
import pandas as pd
from abc import ABC, abstractmethod
from datetime import date, datetime
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as clr
import numpy as np
import pickle
import json

# ====================

class FormatService:
    
    @classmethod
    def normalize(cls, text: str) -> str:
        return '_'.join(text.strip().split())
    
    @classmethod
    def normalize_lower(cls, text: str) -> str:
        return '_'.join(text.strip().lower().split())
    
    @classmethod
    def today(cls) -> str: 
        return date.today().strftime("%Y.%m.%d")
    
    @classmethod
    def now(cls) -> str: 
        return datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    
    @classmethod
    def normalize_directory_path(cls, text: str) -> str:
        text = cls.normalize(text)
        if text[-1]!='/':
            text = text + '/'
        return text
        
    

# ====================    
    
class FileService:
    
    @classmethod
    def create_folder(cls, folder: str) -> None:
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    
# ====================    
    
class FigureService:
    
    @classmethod
    def get_significance_symbol(cls, pvalue, oneStar=0.05, twoStars=0.01, threeStars=0.001):
        symbol = ''
        if (pvalue<=oneStar):
            symbol = '*'
        if (pvalue<=twoStars):
            symbol = '**'
        if (pvalue<=threeStars):
            symbol = '***'
        return symbol

    @classmethod
    def create_font_sizes(cls, regular=20):
        medium = 0.8 * regular
        small = 0.7 * regular
        tiny = 0.6 * regular
        return regular, medium, small, tiny
    
    @classmethod
    def create_arial_narrow_font(cls):
        mpl.rcParams['font.family'] = 'Arial'
        mpl.rc('font',family='Arial')
        return {'fontname':'Arial', 'stretch' : 'condensed'}
    
    @classmethod
    def create_arial_font(cls):
        mpl.rcParams['font.family'] = 'Arial'
        mpl.rc('font',family='Arial')
        return {'fontname':'Arial'}
    
    @classmethod
    def save_fig_with_resolution(cls, fig, output_dir, file_prefix, dpi=100, ext='png'):
        FileService.create_folder(output_dir)
        filename = output_dir + file_prefix + '.' + ext
        fig.savefig(filename, dpi=dpi, format=ext, bbox_inches='tight', orientation='portrait')
    
    @classmethod    
    def extract_colors_from_colormap(cls, n=10, colormap='jet'):
        cmap = cm.get_cmap(colormap)
        norm = mpl.colors.Normalize(vmin=0, vmax=n-1) 
        return [cmap(norm(ind)) for ind in range(n)] 
    
    @classmethod
    def generate_colors_from_colormap(cls, values, colormap='jet', vmin=None, vmax=None):
        cmap = cm.get_cmap(colormap)
        if (vmin is None):
            vmin = min(values)
        if (vmax is None):
            vmax = max(values)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) 
        return [cmap(norm(v)) for v in values]  
    
    @classmethod
    def create_custom_colormap(cls, palette='white', n_segments=256, list_colors=None):
        if list_colors is None:
            if palette=='black':
                list_colors = ['cyan', 'royalblue', 'black', 'crimson', 'pink']
            else:
                list_colors = ['royalblue', 'cyan', 'azure', 'whitesmoke', 'lavenderblush', 'pink', 'crimson']
        return clr.LinearSegmentedColormap.from_list('custom', list_colors, N=n_segments)
    
    @classmethod
    def create_boxplot_options(cls):
        boxprops = dict(linestyle='-', linewidth=0.75, color='black')
        flierprops = dict(marker='o', markersize=4, markeredgewidth=0.3, markeredgecolor='black')
        medianprops = dict(linestyle='-', linewidth=0.75, color='black')
        meanprops = dict(linestyle='-', linewidth=1.5, color='black')
        capprops = dict(color='black', linewidth=0.75)
        whiskerprops = dict(linestyle='--', linewidth=0.75, color='black')            
        boxplot_options = {
            'flierprops': flierprops, 
            'medianprops': medianprops, 
            'meanprops': meanprops, 
            'meanline': True, 
            'showmeans': True,
            'boxprops': boxprops, 
            'capprops': capprops, 
            'whiskerprops': whiskerprops,
            'patch_artist': True
            }
        return boxplot_options
 

# ====================    
    
class SurvivalService:
    
    @classmethod
    def get_survival_name(cls, survival_type: str) -> str:
        if survival_type=='dfs':
            return 'Disease-free survival'
        return 'Overall survival'
    
    @classmethod
    def get_combination_group_name(cls, combination_groups, n) -> str:
        """combination_groups = {'P1': [0, 2], 'P2': [3, 3], 'P3': [4-5]}"""
        for k, v in combination_groups.items(): 
            if (n>=v[0] and n<=v[1]):
                return k
        return np.nan    
           
    @classmethod
    def get_combination_group_names(cls, combination_groups) -> dict:
        combination_group_names = {}
        for k, v in combination_groups.items():
            if v[0]==v[1]:
                combination_group_names[k] = f"{v[0]}"
            else:
                combination_group_names[k] = f"{v[0]}-{v[1]}"
        return combination_group_names
    
    
# ==============================

class Transformer(ABC):
    """Interface Data Transformer"""
 
    @abstractmethod
    def transform(self) -> pd.DataFrame:
        ...

# ==============================

class FeatureReducer(Transformer):
    """Reduce features (columns) to a given list"""

    def __init__(self, data: pd.DataFrame, features: list):
        self.data = data
        self.features = features
    
    def transform(self) -> pd.DataFrame:
        if self.features:
            available_features = sorted(list(set(self.data.columns).intersection(set(self.features))))
            return self.data[available_features]
        return self.data
    
# ==============================

class SampleReducer(Transformer):
    """Reduce samples (index) to a given list"""
    
    def __init__(self, data: pd.DataFrame, samples: list[str]):
        self.data = data
        self.samples = samples
    
    def transform(self) -> pd.DataFrame:
        if self.samples:
            available_samples = sorted(list(set(self.data.index).intersection(set(self.samples))))
            return self.data.loc[available_samples]
        return self.data

# ==============================

class LatexService():
    
    @classmethod
    def df_to_tabular(cls, df: pd.DataFrame):
        text = '\\begin{center}'
        text = text + '\n' + '\\begin{tabular}{|'
        for col in df.columns:
            text = text + 'l|'
        text = text + '}'
        text = text + '\n' + '\\hline' + '\n'
        for col in df.columns:
            normalized_col = str(col).replace('_', '\_').replace('%', '\%')
            text = text + '\\textbf{' + normalized_col + '}' + ' & '
        text = text[0:-3] + ' \\\\' 
        for row in df.index:
            text = text + '\n' + '\\hline' + '\n'
            for col in df.columns:
                normalized_col = str(df.loc[row, col]).replace('_', '\_').replace('%', '\%')
                text = text + normalized_col + ' & '
            text = text[0:-3] + ' \\\\'    
        text = text + '\n' + '\\hline'    
        text = text + '\n' + '\\end{tabular}'   
        text = text + '\n' + '\\end{center}'

        return text

# ==============================