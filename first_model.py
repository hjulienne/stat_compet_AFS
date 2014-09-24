# kind of silly model where we sum up spectra by integrating them
# We try to see if the amplitude of spectrum are informative 

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt 

fi = "./data/training_other.csv"
data_soil = pd.read_csv(fi, sep= '\t')

fi_spec = "./data/training_spectra.csv"
spec = pd.read_csv(fi_spec, sep='\t')

spec = spec.sum(1) # oh this is silly! we lose so much info!!
spec = (spec - spec.mean())/ spec.std() # we just keep the amplitude of spectrums

data_soil['spec_amp'] = spec  
Nr = data_soil.shape[0]
# plot a random sample of data : 

samp = rd.random_integers(0, Nr, 500)
#######draw features #########:
scat_feat = scatter_matrix(data_soil.ix[samp, 1:16], alpha= 0.3, figsize=(15,15), diagonal='kde')
scat_feat.savefig("./plot/scatter_feat.pdf")
scat_feat.close()

scat_target =  scatter_matrix(data_soil.ix[samp, 16:], alpha= 0.3, figsize=(6,6), diagonal='kde')
scat_target.savefig("./plot/scatter_target.pdf")
