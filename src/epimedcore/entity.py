from  src.epimedcore.service import FormatService, FileService, FeatureReducer, SampleReducer
import pandas as pd
import json
import openpyxl
import warnings
from abc import ABC, abstractmethod



# ==============================

class Persistent(ABC):
    """
    Persistent interface:
    Allow to persist an object as a JSON file.
    """

    @abstractmethod
    def dump(self) -> None:
        """Dump the object as a JSON file"""
        ...
    
    @abstractmethod    
    def restore(self) -> None:
        """Restore the object from a JSON file"""
        ...

# ==============================

class Serializable(ABC):            
    """
    Serializable interface:
    Present the object as a dict of primitive types
    in order to prepare it for persistence 
    as a JSON file (see Persistent interface)
    """
    
    def serialize(self):
        """
        Serialize the object as a dict
        containing only primitive types
        """
        serializable_dict = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, Serializable):
                serializable_dict[k] = v.serialize()
            elif isinstance(v, list):
                serializable_list = []
                for vi in v:
                    serializable_list.append(vi.serialize())
                serializable_dict[k] = serializable_list
            elif isinstance(v, dict):
                serializable_sub_dict = dict()
                for ki, vi in v.items():
                    serializable_sub_dict[ki] = vi.serialize()
                serializable_dict[k] = serializable_sub_dict
            else:
                serializable_dict[k] = v
        return serializable_dict


# ==============================

class Entity(Serializable):
    """Abstract Entity"""
    
    def __repr__(self):
        text = ""
        for k, v in self.__dict__.items():
            text = text + f"{k}={v}, "
        return (f"{self.__class__.__name__} [{text[0:-2]}]")


# ==============================

class Project(Entity, Persistent):
    def __init__(self, name: str, root_dir: str, **kwargs) -> None:
        self.name = FormatService.normalize(name)
        self.root_dir = FormatService.normalize_directory_path(root_dir)
        self.project_dir = self.root_dir + name + '/'
        self.data_dir = self.project_dir + 'DATA/'
        self.results_dir = self.project_dir + 'RESULTS/'
        self.figures_dir = self.project_dir + 'FIGURES/'
        self.json_file = self.project_dir + 'project.json'
        self.datasets = dict()
        for k, v in kwargs.items():
            if k.endswith('_dir'):
                v = FormatService.normalize_directory_path(v)
            setattr(self, k, v)
    
    def add_dataset(self, dataset: 'Dataset') -> None:
        self.datasets[dataset.name] = dataset
          
    def print_summary(self):
        print('Project', self.name, self.project_dir)
        for dk, dv in self.datasets.items():
            print('Dataset', dk)
            print('Groups', ', '.join(dv.groups.keys()))        

    def dump(self) -> None:
        folders = []
        for k, v in self.__dict__.items():
            if k.endswith('_dir'):
                folders.append(v)
        for folder in folders:
            FileService.create_folder(folder)
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.serialize(), f, ensure_ascii=True, indent=4)  
     
    def restore(self):
        """Load project from file project.json"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            project_dict = json.load(f)   
        for k, v in project_dict.items():
            if (k!='datasets'):
                setattr(self, k, v)
        self.datasets.clear()
        for dataset_dict in project_dict['datasets'].values():
            dataset = Dataset(**dataset_dict)
            dataset.groups = dict()
            for group_dict in dataset_dict['groups'].values():
                group = Group(**group_dict)
                dataset.add_group(group)
            self.add_dataset(dataset)   

    

# ==============================

class Dataset(Entity):
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.data_filename = None
        self.expgroup_filename = None
        self.groups = dict()
        for k, v in kwargs.items():
            setattr(self, k, v)
        
    def add_group(self, group: 'Group') -> None:
        self.groups[group.name] = group
    
    def add_groups(self, groups: dict[str, 'Group']) -> None:
        self.groups = groups
    
    
# ==============================

class Group(Entity):
    """Group of samples"""
    
    def __init__(self, name: str, **kwargs) -> None:
        self.name = FormatService.normalize(name)
        self.samples = []
        for k, v in kwargs.items():
            setattr(self, k, v)
        
    def serialize(self):
        serializable_dict = dict()
        for k, v in self.__dict__.items():
            serializable_dict[k] = v
        return serializable_dict
    
    def __repr__(self):
        samples_repr = str(self.samples)
        if len(self.samples)>3:
            samples_repr = str(self.samples[0:3]) + '...'
        return (f"{self.__class__.__name__} [name={self.name}, n={len(self.samples)}, samples={samples_repr}]")
        
        
# ==============================

class Study(Entity):
    
    def __init__(self,
                 project: Project, 
                 dataset_name: str,
                 id_genes: list[int] = None,
                 id_gene_col: str = 'id_gene',
                 gene_symbol_col: str = 'gene_symbol',
                 index_type: str = 'id_gene', # 'gene_symbol', 'hybrid'
                 hybrid_symbol: str = '@'
                 ) -> None:
        
        self.project = project
        self.dataset_name = dataset_name
        self.id_genes = id_genes
        self.id_gene_col = id_gene_col
        self.gene_symbol_col = gene_symbol_col
        self.index_type = index_type
        self.hybrid_symbol = hybrid_symbol
        
        self.expgroup = None
        self.parameters = None
        self.data = None  
        self.gene_info = None

    def load_data(self) -> None:
        expgroup_loader = ExpgroupLoader(self)
        expgroup_loader.load()
        parameters_loader = ParametersLoader(self)
        parameters_loader.load()
        data_loader =  DataLoader(self)
        data_loader.load()
        common_samples = set(data_loader.data.index).intersection(set(expgroup_loader.expgroup.index))
        self.expgroup = SampleReducer(expgroup_loader.expgroup, common_samples).transform()
        self.parameters = SampleReducer(parameters_loader.parameters, common_samples).transform()
        self.data = SampleReducer(data_loader.data, common_samples).transform()
        self.gene_info = data_loader.gene_info.loc[self.data.columns]

    def __repr__(self):
        text = f"project={self.project.name}, "
        for k, v in self.__dict__.items():
            if k!='project':
                text = text + f"{k}={v}, "
        return (f"{self.__class__.__name__} [{text[0:-2]}]")


# ==============================

class Loader(ABC):
    """Data loader interface"""
    
    def __init__(self, study: Study) -> None:
        self.study = study
        
    
    def load(self) -> pd.DataFrame:
        ...

# ==============================

class DataLoader(Loader):
    """Load data from a CSV file into a standard pandas DataFrame"""
    
    def __init__(self, study: Study) -> None:
        super().__init__(study)
        self.data = None  
        self.gene_info = None
        self.features = None
    
    def load(self) -> None:
        filename = self.study.project.data_dir + self.study.project.datasets[self.study.dataset_name].data_filename
        data = pd.read_csv(filename, sep=';')
        self._set_id_gene_col_type_integer(data)
        self._extract_gene_info(data)
        self._set_features()
        self._set_data_index(data)
        data = self._drop_gene_info_columns(data)
        data = data.T
        data.index.name = 'id_sample'
        data = FeatureReducer(data, self.features).transform()
        self.data = data
    
    def _set_id_gene_col_type_integer(self, data: pd.DataFrame) -> None:
        has_id_gene = (self.study.id_gene_col in data.columns)
        if has_id_gene:
            data[self.study.id_gene_col] = pd.to_numeric(data[self.study.id_gene_col], downcast='integer')
    
    def _extract_gene_info(self, data: pd.DataFrame) -> None:
        gene_info_cols = []
        has_id_gene = (self.study.id_gene_col in data.columns)
        has_gene_symbol = (self.study.gene_symbol_col in data.columns)
        if has_id_gene:
            gene_info_cols.append(self.study.id_gene_col)
        if has_gene_symbol:
            gene_info_cols.append(self.study.gene_symbol_col)
        self.gene_info = data[gene_info_cols]
        self._set_data_index(self.gene_info)
            
    def _set_data_index(self, data: pd.DataFrame) -> None:
        has_id_gene = (self.study.id_gene_col in data.columns)
        has_gene_symbol = (self.study.gene_symbol_col in data.columns)
        if (self.study.index_type == 'id_gene') and has_id_gene:
            data.index = data[self.study.id_gene_col]
        if (self.study.index_type == 'gene_symbol') and has_gene_symbol:
            data.index = data[self.study.gene_symbol_col]
        if (self.study.index_type == 'hybrid') and has_id_gene and has_gene_symbol:
            data.index = data[self.study.gene_symbol_col] + self.study.hybrid_symbol + data[self.study.id_gene_col].apply('{:.0f}'.format)
            data.index.name = 'id_gene'
    
    def _drop_gene_info_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=self.gene_info.columns)    
    
    def _set_features(self):
        if self.study.id_genes is None:
            self.features = list(self.gene_info.index)
        else:
            self.features = list(self.gene_info[self.gene_info[self.study.id_gene_col].isin(self.study.id_genes)].index)

# ==============================

class ExpgroupLoader(Loader):
    """Load expgroup from a XLSX file into a standard pandas DataFrame"""
    
    def __init__(self, study: Study) -> None:
        super().__init__(study)
        self.expgroup = None  

    def load(self) -> None:
        filename = self.study.project.data_dir + self.study.project.datasets[self.study.dataset_name].expgroup_filename
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.expgroup = pd.read_excel(filename, engine="openpyxl", sheet_name=0, index_col='id_sample')


# ==============================

class ParametersLoader(Loader):
    """Load parameters tab from a XLSX file into a standard pandas DataFrame"""
    
    def __init__(self, study: Study) -> None:
        super().__init__(study)
        self.parameters = None  

    def load(self) -> None:
        filename = self.study.project.data_dir + self.study.project.datasets[self.study.dataset_name].expgroup_filename
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self.parameters = pd.read_excel(filename, engine="openpyxl", sheet_name=1, index_col='id_sample')