"""
separate the spectrum from other kind of data
new features (features engineering is done here)

"""
#import numpy as np
import pandas as pd
import numpy as np
import sys
from features_lib import spec_amp

fn = sys.argv[1]+".csv"

dat = pd.read_csv(fn)
fo1= sys.argv[1]+"_spectra.csv"
spectra = dat.ix[:, 1:3579]

spectra.columns = map(lambda l: l.split('m')[1], spectra.columns)
spectra.columns = spectra.columns.astype('float')
spectra.to_csv(fo1, index=False, sep='\t')

fo2= sys.argv[1]+"_other.csv"
ids = [0] + range(3579, dat.shape[1])
dat_other = dat.ix[:, ids]
#add the integral of spectra as another centered normalized features:
dat_other['spec_amp'] = spec_amp(spectra)
#convert Depth as a dummy variable
dat_other['Depth'] = dat_other['Depth'].map({'Topsoil' : 0, 'Subsoil':1}).astype('int64')

if sys.argv[1]=="./data/training":
#Apply a log transformation on P and Ca
    dat_other['Ca_log10p2'] = np.log10(2+dat_other["Ca"])
    dat_other['P_log10log10p2'] = np.log10(np.log10(2+dat_other["P"]))
    dat_other['SOC_log10p2'] = np.log10(2+dat_other["SOC"])


#write data with the new features:
dat_other.to_csv(fo2, sep='\t', index=False)
