#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 21:08:55 2018

@author: brianhumphreys
"""

#import flint as ft
import numpy as np

'''
These are the functions that I personally wrote and tested 
'''
#inputs a negative number and modulus p. Converts negative to corresponding positive value in mod p space
def convert_neg(neg, p):
    if neg < 0:
        neg = abs(neg)
        neg = neg % p
        return p - neg

#############################################
#            MATRICIES AAAAAAA              #
#############################################


#Inputs : Numpy matrix - A, integer modulus p 
#Output : Numpy matrix - A^-1 (A inverse)
def create_inverse(in_matrix, p):
    #returns (m, n) tuple of matrix with (m x n) dimmensions
    x_dim, y_dim = in_matrix.shape

    #checks if the matrix is square, if not, exit
    if x_dim != y_dim:
        raise Exception("Non-square matrix is non-invertible")

    #real code to create inverse
    else:
        identity_mat = np.identity(x_dim)
        print(identity_mat)
        
    #perform RREF operations in a modular space
    for x in range(x_dim):
        #find mod inverse to 
        scalar = modinv(in_matrix[x][x], p)
        
        in_matrix[x] = (in_matrix[x] * scalar) % p
        identity_mat[x] = (inentity_mat[x] * scalar) % p
        
        for y in range(y_dim):
        
        
        print(in_matrix)


#def perform_operation()
    

'''
Stack overflow functions that implement basic number theory programs
'''

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)
    

def modinv(a, m):
    if a < 0:
        a = a + m
    g, x, y = egcd(a, m)
    if g != 1:
        print ("g is : ", g)
        raise Exception('modular inverse does not exist')
    else:
        return x % m


def matrix_cofactor(matrix):
    C = np.zeros(matrix.shape)
    nrows, ncols = C.shape
    for row in range(nrows):
        for col in range(ncols):
            minor = matrix[np.array(list(range(row))+list(range(row+1,nrows)))[:,np.newaxis],
                           np.array(list(range(col))+list(range(col+1,ncols)))]
            C[row, col] = (-1)**(row+col) * np.linalg.det(minor)
    return C

### ---------- TESTING ---------- ###

test = convert_neg(-20, 101)
modinv(-20, 101)
modinv(test, 101)

my_mat = np.array([[5, 3],
                   [-3, 5]])

create_inverse(my_mat, 101)



'''
matrix = np.array([[5, 3], [-3, 5]])

matrix_cofactor = matrix_cofactor(matrix)
print("cofactor: ", matrix_cofactor)
print("determinant: ", np.linalg.det(matrix))
print("det inv: ", modinv(np.linalg.det(matrix), 101))

inverse = modinv(np.linalg.det(matrix), 101) * matrix_cofactor

print(inverse)'''

