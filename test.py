import LPSM as LPSM
import numpy as np

LPSM.LPSM('megathingy.csv',[-85.292466, 36.920242],[-84.7906, 29.1957],[(-85.5, -84), (28, 38)],(1820,1108),20,np.rad2deg(np.asin(.08)),PSR_travel=False,mode = 'multi', Plot_Legend=True)
