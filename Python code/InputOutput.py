# -*- coding: utf-8 -*-
import csv as csv
import numpy as np
import matplotlib.pyplot as plt

def ReadCSV(fileLocation):
    '''
    Function to read the CSV data files, input is the name of the CSV file, output nparray of vectors.
    
    (string) -> nparray of dimension # objects in dataset * length of each vector in dataset.
    '''
    with open(fileLocation,'r') as csvFile:
        array = []
        reader = csv.reader(csvFile)

        for row in reader:
            array.append(row)
            
        return np.array(array).astype(np.double)
    
def PlotLattice(array, size, save = False, fileName = ""):
    '''
    Function to plot the size x size vector of a dataset as a digit in a matplotlib heatplot.
    
    (list of length size x size, size) -> void.
    '''
    plt.imshow(array.reshape((size,size)), cmap='Greys', interpolation='nearest')
    if save:
        plt.savefig(fileName + ".pdf")
    plt.show()

