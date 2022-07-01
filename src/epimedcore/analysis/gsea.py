from src.epimedcore.entity import Project
from src.epimedcore.analysis import Analysis
from src.epimedcore.analysis.stats import BenjaminiHochberg
from src.epimedcore.service import FileService, FormatService
import json
import gseapy as gp
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp


# ============================== 

class GseaPrerank(Analysis):
    
    def __init__(self,
                 project: Project,
                 rnk: pd.DataFrame = None,
                 genesets: dict[str, list] = dict(),
                 analysis_name: str = 'GSEA_Prerank',
                 prerank_options: dict = dict(),
                 force_name = None
                 ) -> None:
        
        
        self.project = project
        self.rnk = rnk
        self.genesets = genesets
        self.analysis_name = analysis_name
        self.prerank_options = prerank_options
        self.force_name = force_name
        
        self.prerank_options['outdir'] = self.results_dir
        self.json_key='geneSymbols'
        
        
        self.results = pd.DataFrame()
        self.output = None
        
        self._init_data()
        
           
    def import_genesets_from_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            geneset_dict = json.load(f)   
        for k, v in geneset_dict.items():
            self.genesets[k] = v[self.json_key]
            
    def perform(self):
        self.rnk.columns = ['gene_name', 'metric']
        self.rnk = self.rnk.dropna()
        self.rnk = self.rnk.sort_values(by='metric', ascending=False)
        self.rnk = self.rnk.reset_index(drop=True)
        self.output = gp.prerank(rnk=self.rnk, gene_sets=self.genesets, **self.prerank_options)
        self.results= self.output.res2d
    
    def _init_data(self):
        if self.rnk is None:
            self.rnk = pd.DataFrame()
        if self.genesets is None:
            self.genesets = dict()
    
    
    @property 
    def name(self):
        if self.force_name:
            return self.force_name
        return "GSEA Prerank"
    
    @property
    def local_dir(self):
        local_dir = self.name + '/' + FormatService.today() + ' ' + self.analysis_name
        local_dir = local_dir.strip() + '/'
        return local_dir
    
    @property
    def results_dir(self):
        return self.project_results_dir + self.local_dir
    
    @property
    def figures_dir(self):
        return self.project_figures_dir + self.local_dir
    
    @property
    def project_results_dir(self):
        return self.project.results_dir
    
    @property
    def project_figures_dir(self):
        return self.project.figures_dir
        
    @property
    def project_name(self) -> str:
        return self.project.name
    
    @property
    def groups(self) -> dict[str, list[str]]:
        return None
    
    @property
    def dataset_name(self) -> str:
        return None
    
    @property
    def threshold_name(self) -> str:
        return None
    
    @property
    def expgroup(self) -> pd.DataFrame:
        return None
    
    @property
    def parameters(self) -> pd.DataFrame:
        return None
    
    def as_dict(self) -> dict:
        as_dict = dict()
        as_dict['class'] = self.__class__.__name__
        as_dict['name'] = self.name
        as_dict['results_dir'] = self.results_dir
        as_dict['project_name'] = self.project_name
        as_dict['nb_genesets'] = len(self.genesets)
        as_dict['prerank_options'] = self.prerank_options
        sorted_results = self.results.sort_values(by=['fdr', 'nes'], ascending=[True, False])
        top_genesets = list(sorted_results.head(100).index)
        if len(self.genesets)<100:
            as_dict['genesets'] = top_genesets
        else:
            as_dict['genesets'] = [*top_genesets, '...']
        return as_dict
    
    def save_results(self):
        FileService.create_folder(self.results_dir)
        output_file = self.results_dir + 'GSEA_prerank_results.csv'
        selected_columns = ['es', 'nes', 'pval', 'fdr', 'geneset_size', 'matched_size']
        sorted_results = self.results[selected_columns].sort_values(by=['fdr', 'nes'], ascending=[True, False])
        sorted_results.to_csv(output_file, sep=';', index=True, float_format='%.6f')
    
    def __repr__(self):
        return (f"{self.__class__.__name__} ["
                f"name = {self.name}, "
                f"nb_genesets = {len(self.genesets)}, "
                f"rnk = {self.rnk.shape}"
                f"]")
        

# ============================== 

class GseaStats(Analysis):
    
    def __init__(self,
                 project: Project,
                 ranked_data: dict[str, pd.DataFrame] = None,
                 genesets: dict[str, list] = dict(),
                 analysis_name: str = 'GSEA_Stats',
                 force_name = None
                 ) -> None:
        
        self.project = project
        self.ranked_data = ranked_data
        self.genesets = genesets
        self.analysis_name = analysis_name
        self.force_name = force_name
        
        self.results = pd.DataFrame()
        
    def perform(self):
        
        for rnk_name, rnk in self.ranked_data.items():
            rnk.columns = ['gene_name', 'metric']
            rnk = rnk.dropna()
            rnk = rnk.sort_values(by='metric', ascending=True)
            rnk['rank'] = [n for n in range(rnk.shape[0])]
            rnk.index = rnk['gene_name']
            
            for geneset_name, geneset_elements in self.genesets.items():
                idx = geneset_name + '_' + rnk_name 
                geneset_size = len(geneset_elements)
                common_genes = set(rnk['gene_name']).intersection(geneset_elements)
                matched_size = len(common_genes)
                pval_mw = 1.0
                pval_ks = 1.0
                if matched_size>5:
                    rnk_inside = rnk[rnk['gene_name'].isin(common_genes)]
                    rnk_outside = rnk[~rnk['gene_name'].isin(common_genes)]
                    diff_median = rnk_inside['rank'].median() - rnk['rank'].median()
                    direction = 'pos' if diff_median>0 else 'neg'
                    alternative = 'greater' if diff_median>0 else 'less'
                    stat_mw, pval_mw = mannwhitneyu(rnk_inside['rank'], rnk_outside['rank'], alternative=alternative)
                    stat_ks, pval_ks = ks_2samp(rnk_outside['rank'], rnk_inside['rank'], alternative=alternative)
                    self.results.loc[idx, 'geneset'] = geneset_name
                    self.results.loc[idx, 'analysis'] = rnk_name
                    self.results.loc[idx, 'geneset_size'] = geneset_size
                    self.results.loc[idx, 'matched_size'] = matched_size
                    self.results.loc[idx, 'direction'] = direction
                    self.results.loc[idx, 'pval_mw'] = pval_mw
                    self.results.loc[idx, 'pval_ks'] = pval_ks
                    self.results.loc[idx, 'pval'] = max([pval_mw, pval_ks])
        self.results['fdr_mw'] = BenjaminiHochberg().calculate(self.results['pval_mw'])
        self.results['fdr_ks'] = BenjaminiHochberg().calculate(self.results['pval_ks'])  
        self.results['fdr'] = BenjaminiHochberg().calculate(self.results['pval'])           
                
    @property 
    def name(self):
        if self.force_name:
            return self.force_name
        return "GSEA Stats"
    
    @property
    def local_dir(self):
        local_dir = self.name + '/' + FormatService.today() + ' ' + self.analysis_name
        local_dir = local_dir.strip() + '/'
        return local_dir
    
    @property
    def results_dir(self):
        return self.project_results_dir + self.local_dir
    
    @property
    def figures_dir(self):
        return self.project_figures_dir + self.local_dir
    
    @property
    def project_results_dir(self):
        return self.project.results_dir
    
    @property
    def project_figures_dir(self):
        return self.project.figures_dir
        
    @property
    def project_name(self) -> str:
        return self.project.name
    
    @property
    def groups(self) -> dict[str, list[str]]:
        return None
    
    @property
    def dataset_name(self) -> str:
        return None
    
    @property
    def threshold_name(self) -> str:
        return None
    
    @property
    def expgroup(self) -> pd.DataFrame:
        return None
    
    @property
    def parameters(self) -> pd.DataFrame:
        return None
    
    def as_dict(self) -> dict:
        as_dict = dict()
        as_dict['class'] = self.__class__.__name__
        as_dict['name'] = self.name
        as_dict['results_dir'] = self.results_dir
        as_dict['project_name'] = self.project_name
        as_dict['nb_genesets'] = len(self.genesets)
        as_dict['nb_ranked_data'] = len(self.ranked_data)
        return as_dict
    
    def save_results(self):
        FileService.create_folder(self.results_dir)
        output_file = self.results_dir + 'GSEA_stats_results.csv'
        sorted_results = self.results.sort_values(by=['analysis', 'direction', 'fdr'], ascending=[True, False, True])
        sorted_results.to_csv(output_file, sep=';', index=False, float_format='%.6f')
    
    def __repr__(self):
        return (f"{self.__class__.__name__} ["
                f"name = {self.name}, "
                f"nb_genesets = {len(self.genesets)}, "
                f"nb_ranked_data = {len(self.ranked_data)}"
                f"]")