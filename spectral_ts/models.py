import numpy as np
import pandas as pd
from dataclasses import dataclass
from numpy.linalg import eigvals
from numpy.random import normal, uniform
from scipy.linalg import block_diag

'''
Note: If the spectral radius of the coeff matrix is greater than 1, decrease 'eig_max' paramater. 

Sources: 
VARMA Models - https://www.le.ac.uk/users/dsgp1/COURSES/THIRDMET/MYLECTURES/10MULTARMA.pdf
Matrix Polynomials - https://academiccommons.columbia.edu/doi/10.7916/D8805CQC
'''

@dataclass
class VARMA:
    p: int # AR(p) model
    q: int # MA(q) model
    d: int # Number of Series
    N: int # Length of Series
    lmbda: float # sparsity
    eig_max: float # max(eigval) 
        
    def __repr__(self):
        return f"VARMA({self.p},{self.q}) model\n"
    
    def __post_init__(self):
        '''
        creating lambda functions to compute compute VARMA(p,q) in companion form
        '''  
        if self.p > 0:
            self.AR = self.make_coeffs(self.p)
            self.ar_func = lambda x: self.tensor_product(self.AR[:self.d], x)
        else:
            self.ar_func = lambda x: np.zeros(self.d)
        if self.q > 0:
            self.MA = self.make_coeffs(self.q)
            self.ma_func = lambda x: self.tensor_product(self.MA[:self.d], x)
        else:
            self.ma_func = lambda x: np.zeros(self.d)
    
    def spectral_radius(self, matrix):
        return max(abs(eigvals(matrix)))
    
    def make_matrix(self):
        '''
        Creates random diagonal matrix D with entries sampled uniformly from complex unit cirlce
        Creates random matrix P with complex entries from N(0, 1).
        outputs : A = PDP^(-1) 
        
        Note: These serve as the solvents of the matrix polynomial.
        X_t = A_1*x_{t-1} + A_1*x_{t-2} + ...
        '''
        D = np.diag(self.eig_max*np.exp(1j*uniform(0, 2*np.pi, self.d)))
        P = normal(0,1)*np.exp(1j*uniform(0, 2*np.pi, size = (self.d,self.d)))
        A = P@D@np.linalg.inv(P)
        return A
    
    def block_vandermonde(self, solvents):
        '''
        Matrix polynomial analogue of univariate Vandermonde matrix
        '''
        V = []
        for s in solvents:
            row = [s]
            for i in range(len(solvents)-1):
                row.append(np.matmul(s, row[-1]))
            V.append(np.concatenate(row, axis = 1))
        return np.concatenate(V)
    
    def make_coeffs(self, p_q):
        '''
        Companion matrix, C, is diagonalizable such that C = VDV^(-1) 
        V is the block vandermonde matrix corresponding to solvents
        D is a block diagonal of solvents
        '''
        solvents = [self.make_matrix() for i in range(p_q)]
        D = block_diag(*solvents)
        V = self.block_vandermonde(solvents)
        C = np.linalg.inv(V)@D@V
        C = np.flip(C).T.real
        C[abs(C) < self.lmbda] = 0
        return C
    
    def tensor_product(self, A, x):
        A1 = A.reshape(self.d, len(x), self.d)
        return np.tensordot(A1, x)

    def simulate(self):

        print("AR Spectral Radius =", self.spectral_radius(self.AR))
        print("MA Spectral Radius =", self.spectral_radius(self.MA))
        
        # fixing initial conditions and simulating the VARMA process
        x_hist = [normal(0, 1, self.d) for i in range(self.p)]
        e_hist = [normal(0, 1, self.d) for i in range(self.q)]
        for i in range(self.N):
            e_t = normal(0, 1, self.d)
            ar = self.ar_func(x_hist[:-self.p-1:-1])
            ma = self.ma_func(e_hist[:-self.q-1:-1])
            x_t = ar + ma + e_t
            x_hist.append(x_t)
            e_hist.append(e_t)
            
        # getting rid of starting values
        del x_hist[:self.p]
        self.data = pd.DataFrame(x_hist, columns = ['x{}'.format(i) for i in range(1, self.d+1)])