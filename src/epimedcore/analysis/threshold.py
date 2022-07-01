from abc import ABC, abstractmethod
import pandas as pd
from src.epimedcore.service import FormatService

# ==============================

class Threshold(ABC):
    """Interface Threshold"""
    
    @abstractmethod    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def name(self) -> str:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__} [name={self.name}]"


# ==============================

class CustomThreshold:
    
    def __init__(self, name: str, threshold_values: pd.Series) -> None:
        self.name_ = name
        self.threshold_values = threshold_values
        
    def name(self):
        return FormatService.normalize(self.name_)
        
    def calculate(self, data: pd.DataFrame = None) -> pd.Series:
        return self.threshold_values


# ==============================
        
class MeanTreshold(Threshold):
    """Mean threshold"""
    
    def name(self):
        return 'mean_threshold'
     
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data.mean()

# ==============================

class MeanNStdThreshold(Threshold):
    """Mean + N * std threshold""" 
    
    def __init__(self, n_std: float = 2) -> None:
        self.n_std = n_std
        
    def name(self):
        return f"m{self.n_std}sd"
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data.mean() + self.n_std * data.std()

# ==============================

class PercentileThreshold(Threshold):        
    """
    Percentile threshold
        percentile_value: float, fixed percentile to apply for all genes
        percentile_series: pd.Series, series (index=gene) of percentiles for each gene
    """
    
    def __init__(self, percentile, name:str=None) -> None:
        self.percentile = percentile # float or pd.Series
        self._name = name
        
    def name(self):
        if self._name:
            return self._name
        if isinstance(self.percentile, float) or isinstance(self.percentile, int):
            if (self.percentile==50):
                return "median_threshold"
        return "custom_percentile_threshold"
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if isinstance(self.percentile, pd.Series):
            threshold_values = pd.Series(index=data.columns, dtype=float)
            for feature in data.columns:
                threshold_values[feature] =  data[feature].quantile(self.percentile[feature]/100.0)
            return threshold_values
        return data.quantile(self.percentile/100.0)