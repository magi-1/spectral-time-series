import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache

'''
Sources:
Chapter 7 of Time Series Analysis and Its Applications, Shumway and Stoffer
'''

class fourier:

    def __init__(self, input_data, m):
        # data attributes
        self.N, self.d = input_data.shape
        self.m = m
        # fourier frequencies
        self.num_freqs = int(.5/(1/self.N))
        self.f_freqs = np.linspace(0, .5, self.num_freqs)
        self.L = 2*self.m+1 
        
        # data 
        self.mu_list = input_data.mean()
        self.data = (input_data-self.mu_list).to_numpy()
        self.band = np.linspace(-self.m/self.N, self.m/self.N, self.L)
        self.domain = 2*np.pi*np.linspace(0, self.N, self.N)
        
        # sd_matrices
        self.sd_estimates = self.spectral_density()
        
    def cos_basis(self, w):
        return np.cos(w*self.domain)
    
    def sin_basis(self, w):
        return np.sin(w*self.domain)
    
    def dft(self, w):
        cos = self.cos_basis(w)
        sin = self.sin_basis(w)
        X_c = np.tensordot(self.data.T, cos, axes = 1)
        X_s = np.tensordot(self.data.T, sin, axes = 1)
        return self.N**(-1/2)*(X_c - 1j*X_s).reshape(self.d, 1)
        
    def periodogram(self, w):
        # averages over frequency band with size L = 2*m+1
        I = np.zeros(shape = (self.d, self.d), dtype=complex)
        for m_n in self.band:
            X = self.dft(w+m_n)
            I += X@np.matrix(X).H
        return np.array(I/self.L)
    
    @lru_cache(maxsize=None)
    def spectral_density(self):
        # creates list of spectral density matrices 
        return np.array([self.periodogram(w) for w in self.f_freqs])
    
    def plot_pair(self, i, j):
        # overlays two time series and plots auto-cov and cross-spectrum 
        x = [i for i in range(len(self.data))]
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (14,3))
        ax0.plot(x, self.data[:,i], color = 'k')
        ax0.plot(x, self.data[:,j], color = 'r', alpha = .75)
        ax0.set_title('Observed Data')
        ax0.set_xlabel('t')
        ax1.plot(self.f_freqs, abs(self.sd_estimates[:,i,j]), color = 'k')
        ax1.set_title('Spectral Density Estimate')
        ax1.set_xlabel('freq')
        plt.show()
    