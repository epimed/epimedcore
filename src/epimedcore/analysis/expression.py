from src.epimedcore.analysis import Analysis
from src.epimedcore.analysis.threshold import Threshold 
from src.epimedcore.entity import Study
from src.epimedcore.service import SampleReducer, FeatureReducer, FileService, FormatService
import pandas as pd
import itertools
from abc import abstractmethod


# ==============================

class ExpressionAnalysis(Analysis):

    def __init__(self) -> None:
        self.categorical_data: dict[str, pd.DataFrame] = dict()
        self.continuous_data: dict[str, pd.DataFrame] = dict()
        
    @abstractmethod
    def expgroup(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def parameters(self) -> pd.DataFrame:
        ...

# ==============================        
        
class ExpressionFrequency(ExpressionAnalysis):
    
    def __init__(self,
                 study: Study,
                 threshold: Threshold,
                 ref_group_name: str
                 ) -> None:
        super().__init__()
        self.study = study
        self.threshold = threshold
        self.ref_group_name = ref_group_name
        
        self.threshold_values = None
        self.threshold_percentiles = None
        self.frequency = None
        self.sample_size = dict()
        
    @property 
    def name(self):
        return "Expression Frequency"   
    
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
        return self.study.project.results_dir
    
    @property
    def project_figures_dir(self):
        return self.study.project.figures_dir
        
    @property
    def project_name(self) -> str:
        return self.study.project.name
    
    @property
    def groups(self) -> dict[str, list[str]]:
        return self.study.project.datasets[self.study.dataset_name].groups
    
    @property
    def dataset_name(self) -> str:
        return self.study.dataset_name
    
    @property
    def threshold_name(self) -> str:
        return self.threshold.name()
    
    @property
    def expgroup(self) -> pd.DataFrame:
        return self.study.expgroup
    
    @property
    def parameters(self) -> pd.DataFrame:
        return self.study.parameters
    
    def perform(self):
        self._remove_duplicated_index()
        self._generate_continuous_data()
        self._calculate_threshold()
        self._generate_categorical_data()    
    
    def _remove_duplicated_index(self):
        data_T = self.study.data.T
        data_T = data_T[~data_T.index.duplicated(keep='first')]
        self.study.data = data_T.T
        
    def _generate_continuous_data(self):
        for group_name, group in self.groups.items():
            group_data = SampleReducer(data=self.study.data, samples=group.samples).transform()
            self.continuous_data[group_name] = group_data
            self.sample_size[group_name] = group_data.shape[0]    

    def _generate_categorical_data(self):
        self.frequency = pd.DataFrame(index=self.threshold_values.index)
        self.threshold_percentiles = pd.DataFrame(index=self.threshold_values.index)
        for group_name, group_data in self.continuous_data.items():
            group_expressed = group_data > self.threshold_values.loc[group_data.columns]
            group_frequency = 100.0 * group_expressed.sum() / group_expressed.shape[0]
            self.frequency[group_name] = group_frequency
            self.categorical_data[group_name] = group_expressed.replace({True: 1, False: 0})  
            group_threshold_percentiles = pd.Series(index=self.threshold_values.index, dtype=float)
            for feature in self.threshold_values.index:
                group_threshold_percentiles[feature] = self._get_threshold_percentile(group_data[feature], self.threshold_values[feature])
            self.threshold_percentiles[group_name] = group_threshold_percentiles  
    
                    
    def _calculate_threshold(self) -> None:
        ref_group_samples = self.groups[self.ref_group_name].samples
        ref_data = SampleReducer(data=self.study.data, samples=ref_group_samples).transform()
        self.threshold_values = self.threshold.calculate(ref_data)
        self.threshold_values.name = 'threshold'
        self.threshold_percentiles = pd.Series(index=self.threshold_values.index, dtype=float)
        self.threshold_percentiles.name ='threshold'
        
       

    def _get_threshold_percentile(self, expression_values: pd.Series, threshold_value: float) -> float:
        values = expression_values.sort_values().to_numpy()
        if (len(values)==0.0):
            return 0.0
        n = 0
        for i in range(len(values)):
            if (values[i]<=threshold_value):
                n = n + 1
        return 100.0 * n / len(values)
    
    def save_results(self):
        FileService.create_folder(self.results_dir)
        output_file = self.results_dir + 'individual_expression_frequency.csv'
        self.frequency.to_csv(output_file, sep=';', index=True, float_format='%.2f')
        for group_name, group_expressed in self.categorical_data.items():
            output_file = self.results_dir + 'expressed_samples_in_group_' + group_name + '.csv'
            group_expressed.to_csv(output_file, sep=';', index=True)
        self.threshold_values.to_csv(self.results_dir + 'threshold_values.csv', sep=';', index=True)
        self.threshold_percentiles.to_csv(self.results_dir + 'threshold_percentiles.csv', sep=';', index=True , float_format='%.2f')
        
    
    def as_dict(self) -> dict:
        as_dict = dict()
        as_dict['class'] = self.__class__.__name__
        as_dict['name'] = self.name
        as_dict['results_dir'] = self.results_dir
        as_dict['project_name'] = self.project_name
        as_dict['dataset_name'] = self.dataset_name
        as_dict['threshold_name'] = self.threshold_name
        as_dict['ref_group_name'] = self.ref_group_name
        as_dict['nb_features'] = self.study.data.shape[1]
        as_dict['sample_size'] = self.sample_size
        return as_dict
        
    
    def __repr__(self):
        return (f"{self.__class__.__name__} ["
                f"name = {self.name}, "
                f"project_name = {self.project_name}, "
                f"dataset_name = {self.dataset_name}, "
                f"threshold_name = {self.threshold_name}, "
                f"ref_group_name = {self.ref_group_name}"
                f"]")
        
        
# ==============================


class CombinedExpression(ExpressionAnalysis):
    
    def __init__(self,
                 base_analysis: ExpressionAnalysis, # IndividualExpressionFrequency
                 panel: list = None,
                 ) -> None:
        super().__init__()
        self.base_analysis = base_analysis
        self.panel = panel
    
    @property 
    def name(self):
        return "Combined Expression"   
    
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
        return self.base_analysis.study.project.results_dir
    
    @property
    def project_figures_dir(self):
        return self.base_analysis.study.project.figures_dir
    
    @property
    def project_name(self) -> str:
        return self.base_analysis.study.project.name
    
    @property
    def groups(self) -> dict[str, list[str]]:
        return self.base_analysis.study.project.datasets[self.study.dataset_name].groups
    
    @property
    def dataset_name(self) -> str:
        return self.base_analysis.study.dataset_name
    
    @property
    def threshold_name(self) -> str:
        return self.base_analysis.threshold.name()
    
    @property
    def ref_group_name(self) -> str:
        return self.base_analysis.ref_group_name
    
    @property
    def expgroup(self) -> pd.DataFrame:
        return self.base_analysis.study.expgroup
    
    @property
    def parameters(self) -> pd.DataFrame:
        return self.base_analysis.study.parameters
    
        
    def perform(self):
        self._generate_continuous_data()
        self._generate_categorical_data()
           
    def _generate_continuous_data(self):
        if not self.base_analysis.categorical_data:
            self.base_analysis.perform()
        self._set_panel()
        for group_name, group_data in self.base_analysis.continuous_data.items():
            self.continuous_data[group_name] = FeatureReducer(group_data, self.panel).transform()
    
    def _set_panel(self):
        if self.panel is None:
            if self.base_analysis.continuous_data:
                first_group_name = list(self.base_analysis.continuous_data.keys())[0]
                self.panel = list(self.base_analysis.continuous_data[first_group_name].columns)
    
    def _generate_categorical_data(self):
        for group_name, group_data in self.base_analysis.categorical_data.items():
            binarized_data = FeatureReducer(group_data, self.panel).transform()
            self.categorical_data[group_name] = pd.DataFrame(index=binarized_data.index)
            self.categorical_data[group_name]['panel'] = binarized_data.sum(axis=1)
 
            
    def save_results(self):
        FileService.create_folder(self.results_dir)
        for group_name, group_expressed in self.categorical_data.items():
            output_file = self.results_dir + 'expressed_panel_in_group_' + group_name + '.csv'
            group_expressed.to_csv(output_file, sep=';', index=True)
        
    
    def as_dict(self):
        as_dict = dict()
        as_dict['class'] = self.__class__.__name__
        as_dict['name'] = self.name
        as_dict['panel'] = self.panel
        as_dict['results_dir'] = self.results_dir
        as_dict['base_analysis'] = self.base_analysis.as_dict()
        return as_dict
    
    def __repr__(self):
        return (f"{self.__class__.__name__} ["
                f"name = {self.name}, "
                f"panel = {self.panel}, "
                f"base_analysis = {self.base_analysis}"
                f"]")        
        
        
# ==============================


class TotalCombinations(ExpressionAnalysis):
    
    def __init__(self,
                 base_analysis: ExpressionAnalysis, # IndividualExpressionFrequency
                 panel: list = None,
                 n_top: int = 20
                 ) -> None:
        super().__init__()
        self.base_analysis = base_analysis
        self.panel = panel
        self.n_top = n_top
    
        self.combinations = dict() # id_combination -> combination
        self.top_combinations = dict()
        self.n_combinations = pd.DataFrame()
    
    @property 
    def name(self):
        return "Total combinations"   
    
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
        return self.base_analysis.study.project.results_dir
    
    @property
    def project_figures_dir(self):
        return self.base_analysis.study.project.figures_dir
    
    @property
    def project_name(self) -> str:
        return self.base_analysis.study.project.name
    
    @property
    def groups(self) -> dict[str, list[str]]:
        return self.base_analysis.study.project.datasets[self.study.dataset_name].groups
    
    @property
    def dataset_name(self) -> str:
        return self.base_analysis.study.dataset_name
    
    @property
    def threshold_name(self) -> str:
        return self.base_analysis.threshold.name()
    
    @property
    def ref_group_name(self) -> str:
        return self.base_analysis.ref_group_name
    
    @property
    def expgroup(self) -> pd.DataFrame:
        return self.base_analysis.study.expgroup
    
    @property
    def parameters(self) -> pd.DataFrame:
        return self.base_analysis.study.parameters
    
        
    def perform(self):
        self._set_panel()
        self._init_top_combinations()
        self._perform_combinations()
    
    
    def _init_top_combinations(self):    
        for group_name in self.base_analysis.categorical_data.keys():
            self.top_combinations[group_name] = pd.DataFrame()
            self.top_combinations[group_name].index.name = 'id_combination'
        
    def _perform_combinations(self):
        for n in range(1, len(self.panel)+1, 1):
            combinations = list(itertools.combinations(self.panel, n))
            self.n_combinations.loc[n, 'n_combinations'] = len(combinations)
            print('n =', n, 'combinations', len(combinations))
            result_subsets = pd.DataFrame()
            for i, combination in enumerate(combinations):
                id_combination = f"{n}_{i+1}"
                if i%100==0:
                    print('i =', i, 'id_combination =', id_combination)
                self.combinations[id_combination] = combination
                for group_name, group_data in self.base_analysis.categorical_data.items():
                    binarized_data = FeatureReducer(group_data, combination).transform()
                    n_total = binarized_data.shape[0]
                    combination_activated = binarized_data.sum(axis=1)>0 
                    n_activated = combination_activated.sum()
                    percent_activated = 100.0 * n_activated / n_total
                    result_subsets.loc[id_combination, group_name] = percent_activated     
            for group_name in result_subsets.columns:
                sorted_top_result_subsets = result_subsets.sort_values(by=group_name, ascending=False).head(self.n_top)
                for id_combination in sorted_top_result_subsets.index:
                    self.top_combinations[group_name].loc[id_combination, 'n_features'] = n
                    self.top_combinations[group_name].loc[id_combination, 'frequency'] = sorted_top_result_subsets.loc[id_combination, group_name]
                    self.top_combinations[group_name].loc[id_combination, 'combination'] = ', '.join(sorted(list(self.combinations[id_combination])))   

    
    def _set_panel(self):
        if self.panel is None:
            if self.base_analysis.continuous_data:
                first_group_name = list(self.base_analysis.continuous_data.keys())[0]
                self.panel = list(self.base_analysis.continuous_data[first_group_name].columns)
 
            
    def save_results(self):
        FileService.create_folder(self.results_dir)
        for group_name, group_top_combinations in self.top_combinations.items():
            output_file = self.results_dir + 'combinations_in_group_' + group_name + '.csv'
            group_top_combinations.to_csv(output_file, sep=';', index=True)
        output_file = self.results_dir + 'number_of_combinations.csv'
        self.n_combinations.index_name = 'n_features'
        self.n_combinations.to_csv(output_file, sep=';', index=True)
    
    def as_dict(self):
        as_dict = dict()
        as_dict['class'] = self.__class__.__name__
        as_dict['name'] = self.name
        as_dict['panel'] = self.panel
        as_dict['results_dir'] = self.results_dir
        as_dict['base_analysis'] = self.base_analysis.as_dict()
        return as_dict
    
    def __repr__(self):
        return (f"{self.__class__.__name__} ["
                f"name = {self.name}, "
                f"panel = {self.panel}, "
                f"base_analysis = {self.base_analysis}"
                f"]")        