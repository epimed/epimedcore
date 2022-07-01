from abc import ABC

class Plot(ABC):
    
    default_colormap = 'jet' 
    
    def plot(self):
        ...