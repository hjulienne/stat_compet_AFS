import numpy.random as rd
import matplotlib.pyplot as plt


def draw_spectra(data, nl = 5):
    nr = data.shape[0] #number of rows
    
    if nl > nr:
        nl = nr

    for i in rd.random_integers(0, nr, nl):
        plt.plot(data.columns, data.iloc[i])

    plt.show()
