from src.epimedcore.service import FigureService, SurvivalService, FormatService
from src.epimedcore.plot import Plot
from src.epimedcore.analysis.gsea import GseaPrerank 
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ==============================

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# ==============================

class GseaPlot(Plot):
    
    def __init__(self, 
             fig, 
             gsea_prerank: GseaPrerank,
             geneset_name: str = None,
             **kwargs
             ):
        
        self.fig = fig
        self.gsea_prerank = gsea_prerank
        self.geneset_name = geneset_name
        
        self.font = FigureService.create_arial_narrow_font()
        self.regular = 20
        self.title = None
        
        self.pos_label = 'Pos'
        self.pos_label_color = 'crimson'
        self.neg_label = 'Neg'
        self.neg_label_color = 'darkblue'
        
        self.stat_linewidth = 3
        self.stat_color = '#88C544'
        self.cmap = 'seismic'
        
        self.hits_linewidth = 0.5
        self.hits_color = 'dimgrey'
        
        self.fdr = None
        self.pvalue = None
        self.nes = None
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        
    def plot(self) -> None:
        
        self._init_geneset_name()
        
        regular, medium, small, tiny = FigureService.create_font_sizes(regular=self.regular) 
        
        ax_cmap = self.fig.add_axes([0.1, 0.1, 0.8, 0.1])        
        ax_hits = self.fig.add_axes([0.1, 0.2, 0.8, 0.1])
        ax_stat = self.fig.add_axes([0.1, 0.3, 0.8, 0.6])
        
        if self.title is None:
            self.title = self.geneset_name
        ax_stat.set_title(self.title, fontsize=regular, **self.font)
        
        data = self.gsea_prerank.output.results[self.geneset_name]
        # data.keys = odict_keys(['es', 'nes', 'pval', 'fdr', 'geneset_size', 'matched_size', 'genes', 'ledge_genes', 'RES', 'hit_indices'])
        # print(data.keys())
        x = np.arange(len(data['RES']))
        xmin = min(x)
        xmax = max(x)
        
        
        # === Enrichment score ===
        
        ax_stat.plot(x, data['RES'], linewidth=self.stat_linewidth, color=self.stat_color)
    
        pval_min =  1.0/self.gsea_prerank.prerank_options['permutation_num']
        
        
        if self.pvalue is None:
            self.pvalue = data['pval']
        
        if self.fdr is None:
            self.fdr = data['fdr']
        
        if self.nes is None:
            self.nes = data['nes']
        
        
        nes_label = 'NES = '+ "{:.3f}".format(self.nes)
        fdr_label = 'FDR = '+ "{:.3f}".format(self.fdr) + ' ' + FigureService.get_significance_symbol(self.fdr, oneStar=0.25, twoStars=0.1, threeStars=0.05)
        if (self.fdr<pval_min):
            fdr_label = 'FDR < '+ "{:.3f}".format(pval_min) + ' ' + FigureService.get_significance_symbol(self.fdr, oneStar=0.25, twoStars=0.1, threeStars=0.05)
        fdr_label = fdr_label.strip()
            
        pval_label = 'p-val = '+ "{:.3f}".format(self.pvalue) + ' ' + FigureService.get_significance_symbol(self.pvalue)
        if (self.pvalue<pval_min):
            pval_label = 'p-val < '+ "{:.3f}".format(pval_min) + ' ' + FigureService.get_significance_symbol(self.pvalue)
        pval_label = pval_label.strip()
        
        ax_stat.text(.1, .2, fdr_label, transform=ax_stat.transAxes, fontsize=medium, **self.font)
        ax_stat.text(.1, .3, nes_label, transform=ax_stat.transAxes, fontsize=medium, **self.font)
        ax_stat.text(.1, .1, pval_label, transform=ax_stat.transAxes, fontsize=medium, **self.font)
        
        ax_stat.plot([xmin, xmax], [0.0, 0.0], linewidth=1, color='dimgrey')
        
        ax_stat.set_xticks([])
        ax_stat.set_xticklabels([])
        ax_stat.set_ylabel('Enrichment Score', fontsize=regular, **self.font)
        ax_stat.set_xlim([xmin, xmax])
        
        # === Hits ===
        
        ax_hits.vlines(data['hit_indices'], 0, 1, linewidth=self.hits_linewidth, color = self.hits_color)
        ax_hits.spines['top'].set_visible(False)
        ax_hits.tick_params(axis='both', which='both', 
                        bottom=False, top=False,
                        right=False, left=False, 
                        labelbottom=False, labelleft=False)
        ax_hits.set_xlim([xmin, xmax])
        ax_hits.set_ylim([0, 1])
        
        # === Colormap ===
        
        rank_metric = self.gsea_prerank.rnk.set_index('gene_name')['metric']
        rankings = rank_metric.values
        im_matrix = np.tile(rankings, (2, 1))
        
        ax_cmap.set_xlim([xmin, xmax])
        ax_cmap.imshow(im_matrix, aspect='auto', norm=MidpointNormalize(midpoint=0), cmap=self.cmap, interpolation='none')
        ax_cmap.spines['top'].set_visible(False)
        ax_cmap.tick_params(axis='both', which='both', 
                        bottom=False, top=False,
                        right=False, left=False, 
                        labelbottom=False, labelleft=False)
        
        shift = regular/100.0
        ax_cmap.text(.01, -shift, self.pos_label, color=self.pos_label_color, ha='left', va='top', transform=ax_cmap.transAxes, fontsize=regular, **self.font)
        ax_cmap.text(.99, -shift, self.neg_label, color=self.neg_label_color, ha='right', va='top', transform=ax_cmap.transAxes,  fontsize=regular, **self.font)
        
     
    def _init_geneset_name(self):
        if self.geneset_name is None:
            if self.gsea_prerank.genesets:
                self.geneset_name = list(self.gsea_prerank.genesets.keys())[0]