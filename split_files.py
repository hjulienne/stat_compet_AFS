import numpy as np
import pandas as pd

#separate the spectrum from other kind of data
import sys
fn = sys.argv[1]+".csv"

dat = pd.read_csv(fn)
fo1= sys.argv[1]+"_spectra.csv"
spectra = dat.ix[:, 1:3579]


spectra.columns = map(lambda l: l.split('m')[1], spectra.columns)
spectra.columns = spectra.columns.astype('float')
spectra.to_csv(fo1, sep='\t')

fo2= sys.argv[1]+"_other.csv"
dat_other = dat.ix[:, 3579:]
dat_other.to_csv(fo2, sep='\t')
