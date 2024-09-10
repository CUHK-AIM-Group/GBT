import torch
from torch.autograd import Variable, Function
import numpy as np
import scipy as sp
import scipy.linalg as linalg

class LRRLoss(Function):
    def __init__(self, lambda_=0.25):
        self.lambda_ = lambda_

    @staticmethod
    def forward(self, X, y, lds):

        X = X.cpu().numpy()
        classes = np.unique(y)
        C = classes.size
        N, D = X.shape
        DELTA = 1.
        Obj_c = 0
        dX_c = np.zeros((N, D))
        Obj_all = 0
        dX_all = np.zeros((N,D))
        eigThd = 1e-6

        # compute objective and gradient for first term \sum ||Z_i||_*
        for c in classes:
            A = X[y==c,:] 
            # SVD
            U, S, V = sp.linalg.svd(A, full_matrices = False, lapack_driver='gesvd')
            V = V.T
            nuclear = np.sum(S)
            if nuclear>DELTA:
              Obj_c += nuclear
              r = np.sum(S<eigThd)
              uprod = U[:,0:U.shape[1]-r].dot(V[:,0:V.shape[1]-r].T)
              dX_c[y==c,:] += uprod
            else:
              Obj_c+= DELTA
            
        # compute objective and gradient for secon term ||Z_l||_*
        U, S, V = sp.linalg.svd(X, full_matrices = False, lapack_driver='gesvd')  # all classes
        V = V.T
        Obj_all = np.sum(S)
        r = np.sum(S<eigThd)
        uprod = U[:,0:U.shape[1]-r].dot(V[:,0:V.shape[1]-r].T)
        dX_all = uprod

        # objective
        obj = (Obj_c  - Obj_all)/float(N)*float(lds)
        dX = (dX_c  - dX_all)/float(N)*float(lds) 
        self.dX = torch.FloatTensor(dX)
        return torch.FloatTensor([float(obj)]).cuda()
        
    def backward(self, grad_output):
        return self.dX.cuda(), None