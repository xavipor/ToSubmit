#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 18:38:38 2018

@author: javier
"""
from os import listdir
from random import shuffle
from math import floor
import h5py
import numpy as np 
import pdb
class Bachitazion(object):
    def __init__(self, sizeOfBatch=128,pathT="/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/data/newImages/AllPatchesWithMicrobleedsTrain/patches14/"):
        self.files =shuffle(listdir(pathT))
        self.batchSize = sizeOfBatch	
        self.listTrain =listdir(pathT+"Training/") 
#        self.listTrain =listdir(pathT) 
        self.listEval = listdir(pathT+"Evaluation/")
#        self.listEval = listdir(pathT)
	shuffle(self.listTrain)
	shuffle(self.listEval)
        self.counterT = -1
        self.counterE = -1
        self.number_batchesT =floor(len(self.listTrain)/self.batchSize) 
        self.number_batchesE =floor(len(self.listEval)/self.batchSize) 
        self.myShape = [20,20,16]
        self.Path = pathT
    def nextBatch_T (self):
        self.counterT=self.counterT+1
        #Take care, if counterT or counterE is equal to number_batches, the length is goign to be different,
        #But we can save both cases with the following:
	
        currentBatchSize = len(self.listTrain[self.counterT * self.batchSize:(self.counterT*self.batchSize+self.batchSize)])
        X=np.zeros((currentBatchSize,self.myShape[2],self.myShape[0],self.myShape[1],1))
        Y=np.zeros((currentBatchSize,2))
        
        #If you select a boundary to finish the list bigger than the propper list is fine it will work OK 
        for i,element in enumerate(self.listTrain[self.counterT * self.batchSize:(self.counterT*self.batchSize+self.batchSize)]):
            data_path = self.Path+"Training/"+element
#            data_path = self.Path+element
#            pdb.set_trace()
            aux= np.array(h5py.File(data_path)['patchFlatten'])
            aux2 = np.transpose(aux)
#            aux2 = aux - np.mean(aux)
            patch=aux2.reshape([20,20,16],order='F').copy()
            patch2 = patch.transpose([2,1,0])
#            patch = patch*255.0
#            if "WO" in element:
#                auxY = np.array([1,0])
#            else:
#                auxY = np.array([0,1])
            if "WO" in element:
		auxY = np.array([1,0])
	    else:
		auxY = np.array([0,1])

            X[i,:,:,:,0]=patch2
            Y[i,:]=auxY
#        pdb.set_trace()
        if self.counterT == self.number_batchesT:
#            pdb.set_trace()
            self.counterT= -1
        
        
        return X,Y

    def nextBatch_E (self):
        self.counterE=self.counterE+1
        #Take care, if counterT or counterE is equal to number_batches, the length is goign to be different,
        #But we can save both cases with the following:

        currentBatchSize = len(self.listEval[self.counterE * self.batchSize:(self.counterE*self.batchSize+self.batchSize)])
        X=np.zeros((currentBatchSize,self.myShape[2],self.myShape[0],self.myShape[1],1))
        Y=np.zeros((currentBatchSize,2))
        
        #If you select a boundary to finish the list bigger than the propper list is fine it will work OK 
        for i,element in enumerate(self.listEval[self.counterE * self.batchSize:(self.counterE*self.batchSize+self.batchSize)]):
            data_path = self.Path+"Evaluation/"+element
#            data_path = self.Path+element
            aux= np.array(h5py.File(data_path)['patchFlatten'])
            aux2= np.transpose(aux)
#            aux2 = aux - np.mean(aux)

            patch=aux2.reshape([20,20,16],order='F').copy()
            patch2 = patch.transpose([2,1,0])

#            patch = patch*255.0
            if "WO" in element:
                auxY = np.array([1,0])
            else:
                auxY = np.array([0,1])
            
            X[i,:,:,:,0]=patch2
            Y[i,:]=auxY
            
        if self.counterE == self.number_batchesE:
            self.counterE= -1
        
        
        return X,Y

