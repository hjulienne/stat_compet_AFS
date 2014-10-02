import numpy.random as rd
import matplotlib.pyplot as plt
import pandas as pd

def draw_spectra(data, nl = 5):
    nr = data.shape[0] #number of rows
    
    if nl > nr:
        nl = nr
    for i in rd.random_integers(0, nr, nl):
        plt.plot(data.columns, data.iloc[i])
    plt.show()


fi ="./data/training_spectra.csv"
spec = pd.read_csv(fi, sep= '\t')
spec = spec.iloc[:,1:]
ms = spec.mean(1)

#remove mean by rows
for i in range(spec.shape[0]):
    spec.iloc[i] = spec.iloc[i] - ms[i]

