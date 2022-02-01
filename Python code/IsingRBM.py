# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM

import InputOutput as IO

'''
Import the generated lattices and labels and convert the lattices from {-1,-1} to {0,1} because the RBM expects binary values.
'''
latticesTotal = IO.ReadCSV('Lattices16.csv')
temperatureLabels = IO.ReadCSV('TemperatureLabels16.csv')
latticesTotal = (latticesTotal + 1) / 2

'''
Import the observables as generated using MCMC to plot to compare to the RBM observables.
'''
cMCMC = IO.ReadCSV('data/SpecificHeatPerSpin16.csv')
mMCMC = IO.ReadCSV('data/MagnetisationPerSpin16.csv')
eMCMC = IO.ReadCSV('data/EnergyPerSpin16.csv')
temperatureLabelsMCMC = IO.ReadCSV('data/TemperatureLabelsMCMC16.csv')

cMCMC = [x[0] for x in cMCMC]
mMCMC = [x[0] for x in mMCMC]
eMCMC = [x[0] for x in eMCMC]

plt.plot(temperatureLabelsMCMC, eMCMC, 'bo', markersize = 1)
plt.xlabel(r'$T$ [dimensionless units]',fontsize=18)
plt.ylabel(r'$\langle E \rangle$ [dimensionless units]',fontsize=18)
plt.savefig("eMCMC.pdf")
plt.show()

plt.plot(temperatureLabelsMCMC, np.abs(mMCMC), 'bo', markersize = 1)
plt.xlabel(r'$T$ [dimensionless units]',fontsize=18)
plt.ylabel(r'$\langle E \rangle$ [dimensionless units]',fontsize=18)
plt.savefig("mMCMC.pdf")
plt.show()

plt.plot(temperatureLabelsMCMC, cMCMC, 'bo', markersize = 1)
plt.xlabel(r'$T$ [dimensionless units]',fontsize=18)
plt.ylabel(r'$c$ [dimensionless units]',fontsize=18)
plt.savefig("cMCMC.pdf")
plt.show()

latticeSize = int(len(latticesTotal[0])**(1/2))

'''
Function to calculate the total energy of a lattice at a given size.
'''
def LatticeEnergy(lattice, size):
    Energy = 0
    
    lattice = (2*lattice) - 1
    lattice = np.reshape(lattice, (size,size))
       
    for i in range(size):
        for j in range(size):
            Energy = Energy + -1*lattice[i,j] *( lattice[(i + 1)%size,j] + lattice[(i - 1)%size,j] + lattice[i,(j+1)%size] + lattice[i ,(j-1)%size])

    return Energy

'''
Function to calculate the total magnetisation of a lattice at a given size.
'''
def LatticeMagnetisation(lattice,size):    
    lattice = (2*lattice) - 1
    
    return np.sum(lattice)

print(latticesTotal.shape)

'''
Plot the energy and the magnetisation of each lattice in the dataset to verify whether it is correct.
'''
maglist = []
elist = []

tempList = np.unique(temperatureLabels, return_counts = True)[0]
nItr = np.unique(temperatureLabels, return_counts = True)[1][0]

for lattice in latticesTotal:
    
    #print(lattice)
    maglist.append(np.abs(LatticeMagnetisation(lattice, latticeSize))/(latticeSize**2))
    elist.append(LatticeEnergy(lattice, latticeSize)/(latticeSize**2))
    
plt.plot(maglist, 'bo', markersize = 1)
plt.xlabel('Data points')
plt.ylabel('Magnetization')
plt.savefig("mag.pdf")
plt.show()

plt.plot(elist, 'bo', markersize = 1)
plt.xlabel('Data points')
plt.ylabel('Energy')
plt.savefig("energy.pdf")
plt.show()

'''
Reshape the lattice and label arrays such that each temperature dataset is in a separate vector.
'''
latticesTotal = np.reshape(latticesTotal, (len(tempList), nItr, latticeSize**2))
temperatureLabels = np.reshape(temperatureLabels, (len(tempList),nItr))

magnetisationVariance = []
energyVariance = []

magnetisation = []
energy = []

'''
Define a RBM, train it and use it to generate new lattices. Use these lattices to calculate the observables.
This process is done at each temperature in the range and at each number of hidden nodes that we want to calculate.
'''
numberHiddenNodes = [8,16,256]

for h in range(len(numberHiddenNodes)):
    magnetisationOneTemperature = []
    energyOneTemperature = []
    
    magnetisationVarianceOneTemperature = []
    energyVarianceOneTemperature = []
    
    temperatureCounter = 0
    
    for latticeOneTemperature in latticesTotal:
        temperature = temperatureLabels[temperatureCounter,0]
        print("Temperature: " + str(temperature))
      
        rbm = BernoulliRBM(n_components = numberHiddenNodes[h], learning_rate=0.001 , verbose=False, n_iter = 100, batch_size = 10)                
        rbm.fit(latticeOneTemperature)
           
        '''
        Plot and save the weightmatrix for each temperature.
        '''
        plt.imshow(rbm.components_)
        plt.ylabel("Hidden nodes")
        plt.xlabel("Visible nodes")
        plt.savefig("weightsTemp" + str(temperature) + "Nh" + str(numberHiddenNodes[h]) + ".pdf")
        plt.show()
        
        '''
        Generate a random lattice and use Gibbs sampling to generate new lattices at this temperature. After equilibration sample the system and use those samples to calculate the observables.
        '''
        magnetisationSampleList = []
        energySampleList = []
        
        randomlattice =  np.random.randint(2, size=latticeSize**2)
    
        equilibrationTime = 1000
        numberSamples = 1000
               
        for i in range(equilibrationTime * latticeSize**2 + numberSamples * latticeSize**2):
            lattice = rbm.gibbs(randomlattice)
            
            if i > equilibrationTime * latticeSize**2 and i % latticeSize**2 == 0:           
                magnetisationSampleList.append(LatticeMagnetisation(lattice, latticeSize)/latticeSize**2)
                energySampleList.append(LatticeEnergy(lattice, latticeSize))
                        
        magnetisationOneTemperature.append(np.average(magnetisationSampleList))
        energyOneTemperature.append(np.average(energySampleList)/latticeSize**2)
        
        magnetisationVarianceOneTemperature.append((latticeSize**2/temperature) * np.var(magnetisationSampleList))
        energyVarianceOneTemperature.append((1/(temperature**2 * latticeSize**2)) * np.var(energySampleList))
            
        temperatureCounter = temperatureCounter + 1
        
    magnetisation.append(magnetisationOneTemperature)
    energy.append(energyOneTemperature)
    
    magnetisationVariance.append(magnetisationVarianceOneTemperature)
    energyVariance.append(energyVarianceOneTemperature)

colours = ['ro','go','bo']

for i in range(len(numberHiddenNodes)):
    plt.plot(tempList, energyVariance[i], colours[i])
plt.ylabel("c [dimensionless units]")
plt.xlabel(r'$T$ [dimensionless units]')
plt.legend([r'$N_h = 8$', r'$N_h = 16$', r'$N_h = 256$'], loc='upper right')
plt.savefig("cRBM.pdf")
plt.show()

for i in range(len(numberHiddenNodes)):
    plt.plot(tempList, magnetisation[i], colours[i])
plt.ylabel(r'$\langle M \rangle$ [dimensionless units]')
plt.xlabel(r'$T$ [dimensionless units]')
plt.legend([r'$N_h = 8$', r'$N_h = 16$', r'$N_h = 256$'], loc='upper right')
plt.savefig("magRBM.pdf")
plt.show()

for i in range(len(numberHiddenNodes)):
    plt.plot(tempList, energy[i], colours[i])
plt.ylabel(r'$\langle E \rangle$ [dimensionless units]')
plt.xlabel(r'$T$ [dimensionless units]')
plt.legend([r'$N_h = 8$', r'$N_h = 16$', r'$N_h = 256$'], loc='lower right')
plt.savefig("eRBM.pdf")
plt.show()