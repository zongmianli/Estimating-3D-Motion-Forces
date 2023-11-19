import numpy as np
import numpy.linalg as LA

import pinocchio as se3
from pinocchio.utils import *
from pinocchio.explog import exp,log


def rotation_matrix(a, b):
	'''compute a rotation matrix from vector a to vector b'''
	a = np.matrix(a)
	b = np.matrix(b)
	if a.shape != (3,1):
		a = a.T
	if b.shape != (3,1):
		b = b.T
	v = cross(a, b)
	c = a.T*b # cos of angle, note that the result is a 1x1 matrix
	v_cross = cross_3d(v)
	R = eye(3) + v_cross + v_cross*v_cross*(1/(1+c[0,0]))
	return R


def cross_3d(v):
	'''given 3d vector v this function outputs v_cross matrix'''
	v = np.matrix(v)
	if v.shape != (3,1):
		v = v.T
	v_cross = np.matrix([[0., -v[2,0], v[1,0]],
					 	 [v[2,0], 0., -v[0,0]],
						 [-v[1,0], v[0,0], 0.]])
	return v_cross


def procrustes(A, B):
    '''
    Solves the orthogonal Procrustes problem given a set of 3D points
    A (3 x N) and a set of target 3D points B (3 x N). Namely, it
    computes a group of R(otation), t(ranslation) and s(cale) that
    aligns A with B.
    '''
    transposed = False
    if A.shape[0]!=3:
        A = A.T
        B = B.T
        transposed = True
    N = A.shape[1]
    assert(B.shape==(3,N))
    # compute mean
    a_bar = A.mean(axis=1, keepdims=True)
    b_bar = B.mean(axis=1, keepdims=True)
    # calculate rotation
    A_c = A - a_bar
    B_c = B - b_bar
    M = A_c.dot(B_c.T)
    U, Sigma, Vh = LA.svd(M)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1,-1] = LA.det(V)*LA.det(U)
    R = V.dot(Z.dot(U.T))
    # compute scale
    s = np.trace(R.dot(M)) / np.trace(A_c.T.dot(A_c))
    # compute translation
    t = b_bar - s*(R.dot(a_bar))
    # compute A after alignment
    A_hat = s*(R.dot(A)) + t
    if transposed:
        A_hat = A_hat.T
    return (R, t, s, A_hat)
