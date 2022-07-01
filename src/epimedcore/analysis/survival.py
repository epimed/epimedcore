from src.epimedcore.analysis import Analysis
from src.epimedcore.analysis.expression import ExpressionAnalysis
from src.epimedcore.service import FormatService, FileService
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.exceptions import ConvergenceError
import warnings
from abc import ABC, abstractmethod

# ==============================  

class SurvivalAnalysis(Analysis):
    """Abstract survival analysis"""
    
    def __init__(self,
        expression_analysis: ExpressionAnalysis,
        **kwargs
        ) -> None:
        
        self.expression_analysis = expression_analysis
        self.survival_type: str = 'os' # os/dfs
        self.max_survival_time: float = 120
        self.panel = None
        self.group_names = None 
        self.alpha = 0.05

        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self._set_panel()
        self._set_group_names()
        
        self.duration_col = self.survival_type + '_months'
        self.event_col = self.survival_type + '_censor'
        self.results = dict()
        self.sample_size = dict()
        
        
        
    def _set_panel(self):
        if self.panel is None:
            if self.expression_analysis.categorical_data:
                first_group_name = list(self.expression_analysis.categorical_data.keys())[0]
                self.panel = list(self.expression_analysis.categorical_data[first_group_name].columns)
        else:
            if isinstance(self.panel, str) or isinstance(self.panel, int):
                self.panel = [self.panel]

    def _set_group_names(self):
        if self.group_names is None:
            self.group_names = list(self.expression_analysis.categorical_data.keys())

    def generate_survival_data(self, feature, data: pd.DataFrame, expgroup: pd.DataFrame) -> pd.DataFrame:
        survival_data = pd.DataFrame(index=data.index)
        survival_data[feature] = data[feature]
        for id_sample in data.index:
            shifted_time = expgroup.loc[id_sample, self.duration_col]
            shifted_event = expgroup.loc[id_sample, self.event_col]
            if (shifted_time > self.max_survival_time):
                shifted_time = self.max_survival_time
                shifted_event = 0.0
            survival_data.loc[id_sample, 'time'] = shifted_time
            survival_data.loc[id_sample, 'event'] = shifted_event
        survival_data = survival_data.dropna(axis=0)
        return survival_data

    @property
    def local_dir(self):
        local_dir = self.name + '/' + FormatService.today() + ' ' + self.dataset_name  + ' ' + self.threshold_name + ' ' + self.ref_group_name + '/'
        return local_dir

    @property
    def results_dir(self):
        return self.project_results_dir + self.local_dir
    
    @property
    def figures_dir(self):
        return self.project_figures_dir + self.local_dir

    @property
    def project_results_dir(self):
        return self.expression_analysis.project_results_dir
    
    @property
    def project_figures_dir(self):
        return self.expression_analysis.project_figures_dir
        
    @property
    def project_name(self) -> str:
        return self.expression_analysis.project_name
    
    @property
    def groups(self) -> dict[str, list[str]]:
        return self.expression_analysis.groups
    
    @property
    def dataset_name(self) -> str:
        return self.expression_analysis.dataset_name
    
    @property
    def threshold_name(self) -> str:
        return self.expression_analysis.threshold_name
    
    @property
    def ref_group_name(self) -> str:
        return self.expression_analysis.ref_group_name
    
    @property
    def categorical_data(self) -> dict:
        return self.expression_analysis.categorical_data
    
    @property
    def continuous_data(self) -> dict:
        return self.expression_analysis.continuous_data
    
    @property
    def expgroup(self) -> dict:
        return self.expression_analysis.expgroup
    
    def as_dict(self) -> dict:
        as_dict = dict()
        as_dict['class'] = self.__class__.__name__
        as_dict['name'] = self.name
        as_dict['results_dir'] = self.results_dir
        as_dict['survival_type'] = self.survival_type
        as_dict['max_survival_time'] = self.max_survival_time
        as_dict['duration_col'] = self.duration_col
        as_dict['event_col'] = self.event_col
        as_dict['panel'] = self.panel
        as_dict['group_names'] = self.group_names 
        as_dict['alpha'] = self.alpha
        as_dict['sample_size'] = self.sample_size
        as_dict['expression_analysis'] = self.expression_analysis.as_dict()
        return as_dict   

      

# ==============================   

class UnivariateSurvival(SurvivalAnalysis): 
    
    @property
    def name(self) -> str:
        return 'Univariate Survival Analysis'

    def perform(self) -> None:
        cox_pvalues = pd.DataFrame()
        cox_hrs = pd.DataFrame()
        logrank_pvalues = pd.DataFrame()
        logrank_hrs = pd.DataFrame()
        min_pvalues = pd.DataFrame()
        max_pvalues = pd.DataFrame()
        for group_name in self.group_names:
            group_continuous_data = self.expression_analysis.continuous_data[group_name]
            group_categorical_data = self.expression_analysis.categorical_data[group_name]
            expgroup = self.expression_analysis.expgroup.loc[group_categorical_data.index]
            for feature in self.panel:
                survival_continuous_data = self.generate_survival_data(feature, group_continuous_data, expgroup)
                survival_categorical_data = self.generate_survival_data(feature, group_categorical_data, expgroup)
                self._update_sample_size(group_name, survival_categorical_data)
                (cox_pvalue, cox_hr) = UnivariateCox().calculate(feature, survival_continuous_data)
                (logrank_pvalue, logrank_hr) = Logrank().calculate(feature, survival_categorical_data)
                min_pvalue = self._calculate_min_pvalue(cox_pvalue, logrank_pvalue)
                max_pvalue = self._calculate_max_pvalue(cox_pvalue, logrank_pvalue)
                cox_pvalues.loc[feature, group_name] = cox_pvalue
                cox_hrs.loc[feature, group_name] = cox_hr
                logrank_pvalues.loc[feature, group_name] = logrank_pvalue
                logrank_hrs.loc[feature, group_name] = logrank_hr
                min_pvalues.loc[feature, group_name] = min_pvalue
                max_pvalues.loc[feature, group_name] = max_pvalue
        self.results['cox_pvalues'] = cox_pvalues
        self.results['cox_hrs'] = cox_hrs
        self.results['logrank_pvalues'] = logrank_pvalues
        self.results['logrank_hrs'] = logrank_hrs
        self.results['min_pvalues'] = min_pvalues
        self.results['max_pvalues'] = min_pvalues
        self.results['significant_at_least_one_pvalue'] = ((min_pvalues < self.alpha) & (logrank_hrs > 1.0)).replace({True: 1, False: 0})
        self.results['significant_all_pvalues'] = ((max_pvalues < self.alpha) & (logrank_hrs > 1.0)).replace({True: 1, False: 0})
        for v in self.results.values():
            v.index.name = 'panel'

    def _update_sample_size(self, group_name, survival_data):
        feature_sample_size = survival_data.shape[0]
        if group_name not in self.sample_size.keys():
            self.sample_size[group_name] = {'min_sample_size': feature_sample_size, 'max_sample_size': feature_sample_size}
        min_sample_size = self.sample_size[group_name]['min_sample_size']
        max_sample_size = self.sample_size[group_name]['max_sample_size']
        if (feature_sample_size < min_sample_size):
            self.sample_size[group_name]['min_sample_size'] = feature_sample_size
        if (feature_sample_size > max_sample_size):
            self.sample_size[group_name]['max_sample_size'] = feature_sample_size

    
    def _calculate_max_pvalue(self, pvalue1: float, pvalue2: float) -> float:
        max_pvalue = np.nan
        at_meast_one_pvalue_not_nan = ~np.isnan(pvalue1) or ~np.isnan(pvalue2)
        if at_meast_one_pvalue_not_nan :
            max_pvalue = np.nanmax([pvalue1, pvalue2])    
        return max_pvalue
    
    
    def _calculate_min_pvalue(self, pvalue1: float, pvalue2: float) -> float:
        min_pvalue = np.nan
        at_least_one_pvalue_not_nan = ~np.isnan(pvalue1) or ~np.isnan(pvalue2)
        if at_least_one_pvalue_not_nan:
            min_pvalue = np.nanmin([pvalue1, pvalue2])    
        return min_pvalue         
    
    def save_results(self):
        FileService.create_folder(self.results_dir)
        for k, v in self.results.items():
            output_file = self.results_dir + 'survival_analysis_results_' + k + '.csv'
            v.to_csv(output_file, sep=';', index=True)                
    

    def __repr__(self):
        return (f"{self.__class__.__name__} ["
                f"name = {self.name}, "
                f"survival_type = {self.survival_type}, "
                f"max_survival_time = {self.max_survival_time}, "
                f"duration_col = {self.duration_col}, "
                f"event_col = {self.event_col}, "
                f"panel = {self.panel}, "
                f"group_names = {self.group_names}, "
                f"expression_analysis = {self.expression_analysis}"
                f"]")  

# ============================== 

class MultivariateSurvival(SurvivalAnalysis):

    def __init__(self,
        expression_analysis: ExpressionAnalysis,
        continuous_features: list = None,
        categorical_features: list = None,
        expgroup_features: list[str] = None,
        **kwargs
        ) -> None:
        
        super().__init__(expression_analysis, **kwargs)
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.expgroup_features = expgroup_features
        
        self.cox_data = dict() # dict[str, pd.DataFrame]
        self.results = dict() # dict[str, pd.DataFrame]
        self.summary = None        
        self.summary_pvalue_colname = 'p-value'
        self.summary_hr_colname = 'HR'
        
    @property   
    def name(self):
        return 'Multivariate Survival Analysis'
    

    @property
    def local_dir(self):
        list_features = self._get_list_features()
        # local_dir = self.name + '/' + self._get_features_name() + ' ' + ' '.join(list_features) + '/' + FormatService.today() + ' ' + self.dataset_name + '/' 
        local_dir = self.name + '/' + self._get_features_name() + ' ' + str(len(list_features)) + '/' + FormatService.today() + ' ' + self.dataset_name + '/' 
        return local_dir
    
    
    def _get_features_name(self) -> str:
        is_continuous = self.continuous_features is not None
        is_categorical = self.categorical_features is not None
        is_expgroup = self.expgroup_features is not None
        if (is_continuous) and (not is_categorical) and (not is_expgroup):
            return 'Continuous features'
        if (is_categorical) and (not is_continuous) and (not is_expgroup):
            return 'Categorical features'
        if (is_expgroup) and (not is_continuous) and (not is_categorical):
            return 'Expgroup features'
        return 'Mixed features'
    
    def _get_list_features(self) -> list:
        list_features = list()
        category_features = [self.continuous_features, self.categorical_features, self.expgroup_features]
        for features in category_features:
            if features is not None:
                for feature in features:
                    list_features.append(feature)
        return sorted(list_features)
       
    def perform(self):
        self._generate_cox_data()
        self._fit_cox_model()
        self._add_summary()
    
    def _add_summary(self):
        colnames = list()
        for group_name in self.group_names:
            for metrics in [self.summary_pvalue_colname, self.summary_hr_colname]:
                colind = (group_name, metrics)
                colnames.append(colind)
        colnames = pd.MultiIndex.from_tuples(colnames, names=['Group', 'Metrics'])
        self.summary = pd.DataFrame(columns=colnames)
        self.summary.index.name = 'Feature'
        for group_name, result in self.results.items():
            for feature in result.index:
                self.summary.loc[feature, (group_name, self.summary_pvalue_colname)] = '{:.5f}'.format(result.loc[feature, 'p']).replace('nan', 'NA')
                self.summary.loc[feature, (group_name, self.summary_hr_colname)] = '{:.2f}'.format(result.loc[feature, 'exp(coef)']).replace('nan', 'NA')
           
    def _fit_cox_model(self):
        cph =  CoxPHFitter()
        for group_name in self.group_names:
            if not self.cox_data[group_name].empty:
                cox_data = self.cox_data[group_name]
                variance = cox_data.drop(['time', 'event'], axis=1).var(axis=0)
                try:
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        cph.fit(cox_data, duration_col='time', event_col='event', show_progress=False)
                    self.results[group_name] = cph.summary
                except (ConvergenceError) as err: 
                    self.results[group_name] = pd.DataFrame(index=variance.index)
                    self.results[group_name]['p'] = np.nan
                    self.results[group_name]['exp(coef)'] = np.nan
                    self.results[group_name]['Variance'] = variance
                    self.results[group_name]['Error'] = f"{err}"
                    print(f"---> Warning: Cox model convergence error in {group_name} {err=}")
            
    def _generate_cox_data(self):
        expgroup = self.expression_analysis.expgroup
        numerical_variables = list(expgroup.select_dtypes('number').columns)
        for group_name in self.group_names:
            group_continuous_data = self.expression_analysis.continuous_data[group_name]
            group_categorical_data = self.expression_analysis.categorical_data[group_name]      
            id_samples =  group_continuous_data.index
            cox_data = pd.DataFrame(index=id_samples)
            for id_sample in id_samples:
                shifted_time = expgroup.loc[id_sample, self.duration_col]
                shifted_event = expgroup.loc[id_sample, self.event_col]
                if (shifted_time > self.max_survival_time):
                    shifted_time = self.max_survival_time
                    shifted_event = 0.0
                cox_data.loc[id_sample, 'time'] = shifted_time
                cox_data.loc[id_sample, 'event'] = shifted_event
                
            if self.continuous_features is not None:
                for feature in self.continuous_features:
                    cox_data.loc[id_samples, feature] = group_continuous_data.loc[id_samples, feature]
            if self.categorical_features is not None:
                for feature in self.categorical_features:
                    cox_data.loc[id_samples, feature] = group_categorical_data.loc[id_samples, feature]
            if self.expgroup_features is not None:
                for feature in self.expgroup_features:
                    if feature in numerical_variables:
                        cox_data.loc[id_samples, feature] = expgroup.loc[id_samples, feature]
                    else:
                        list_categories = sorted(list(expgroup.loc[id_samples, feature].unique()))
                        dict_categories = dict(zip(list_categories, np.arange(len(list_categories))))
                        encoded_values = [dict_categories[v] for v in list(expgroup.loc[id_samples, feature])]
                        cox_data.loc[id_samples, feature] = encoded_values
            
            cox_data = cox_data.dropna(axis=0)
            self.cox_data[group_name] = cox_data
            
   
    
    def style(self, df):
        style_no_highlight = '' 
        highlight_light_green = 'background-color: #C1FFC1'
        s = pd.DataFrame(style_no_highlight, index=df.index, columns=df.columns)
        for ind in s.index:
            for col in s.columns: 
                if (col[1]==self.summary_pvalue_colname):
                    try:
                        p_value = float(df.loc[ind, col])
                        if (p_value<=self.alpha):
                            s.loc[ind, col] = highlight_light_green
                    except:
                        pass
        return s
    
                    
    def save_results(self):
        FileService.create_folder(self.results_dir)
        
        for k, v in self.results.items():
            output_file = self.results_dir + 'multivariare_survival_analysis_results_' + k + '.csv'
            v.to_csv(output_file, sep=';', index=True)
        
        file_prefix = 'multivariare_survival_analysis_summary_' + self.dataset_name
        writer = pd.ExcelWriter(self.results_dir + file_prefix + '.xlsx', engine='openpyxl')
        self.summary.style.apply(self.style, axis=None).to_excel(writer, sheet_name='Summary')
        writer.save()
        
    
    def as_dict(self) -> dict:
        as_dict = super().as_dict()
        as_dict['continuous_features'] = self.continuous_features
        as_dict['categorical_features'] = self.categorical_features
        as_dict['expgroup_features'] = self.expgroup_features
        return as_dict   
    
    
    
    def __repr__(self):
        return (f"{self.__class__.__name__} ["
                f"name = {self.name}, "
                f"continuous_features = {self.continuous_features}, "
                f"categorical_features = {self.categorical_features}, "
                f"expgroup_features = {self.expgroup_features}, "
                f"survival_type = {self.survival_type}, "
                f"max_survival_time = {self.max_survival_time}, "
                f"group_names = {self.group_names}"
                f"]"
                )
   
# ==============================   

class SurvivalModel(ABC):
    """Abstract survival model"""
        
    def __init__(self):
        self.cph =  CoxPHFitter()
        
    @abstractmethod    
    def calculate(self) -> tuple:
        ...

# ==============================   

class UnivariateCox(SurvivalModel):
        
    def calculate(self, feature, data: pd.DataFrame) -> tuple:
        """
        Calculate Cox model
        Data should contain at least 3 columns: time, event and feature.
        """ 
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self.cph.fit(data, duration_col='time', event_col='event', show_progress=False)
            pvalue = self.cph.summary.p[feature]
            hr = self.cph.summary['exp(coef)'][feature]
            return (pvalue, hr)
        except:
            return (np.nan, np.nan)
    
# ==============================

class Logrank(SurvivalModel):
    
    """
    Calculate Logrank model
    Data should contain at least 3 columns: time, event and feature.
    """
    
    def calculate(self, feature, data: pd.DataFrame) -> tuple:
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                self.cph.fit(data, duration_col='time', event_col='event', show_progress=False)
                logrank = multivariate_logrank_test(data['time'], data[feature], data['event'])
            hr = self.cph.summary['exp(coef)'][feature]
            pvalue = logrank.p_value
            return (pvalue, hr)
        except:
            return (np.nan, np.nan)
    