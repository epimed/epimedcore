from abc import ABC, abstractmethod
import json
import pickle
from src.epimedcore.service import  FileService

# ==============================        

class Analysis(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def results_dir(self):
        ...

    @property
    @abstractmethod
    def figures_dir(self):
        ...

    @property
    @abstractmethod
    def project_results_dir(self):
        ...
        
    @property
    @abstractmethod
    def project_figures_dir(self):
        ...

    @property
    @abstractmethod
    def project_name(self) -> str:
        ...

    @property
    @abstractmethod
    def groups(self) -> dict[str, list[str]]:
        ...

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        ...
    
    @property
    @abstractmethod
    def threshold_name(self) -> str:
        ...

    @abstractmethod
    def perform(self):
        ...

    @abstractmethod
    def as_dict(self) -> dict:
        ...
        
    @abstractmethod
    def save_results(self) -> dict:
        ...
      
    def dump(self):
        self.to_json()
        self.to_pickle()
        
    
    def to_json(self) -> None:
        FileService.create_folder(self.results_dir)
        json_file = self.results_dir + 'analysis.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.as_dict(), f, ensure_ascii=True, indent=4)
    
    def to_pickle(self) -> None:
        FileService.create_folder(self.results_dir)
        pickle_file = self.results_dir + 'analysis.pickle'
        with open(pickle_file, 'bw') as f:
            pickle.dump(self, f)
    
    @classmethod   
    def from_pickle(cls, pickle_file: str):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)