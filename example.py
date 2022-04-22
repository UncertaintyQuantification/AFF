#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:21:08 2022

@author: hao li
"""
import numpy as np
from utils import aff

# load the dataset contains the geometry information and the force-fields

dataset=np.load('./dataset/uracil_dft.npz')


AFF_train=aff.AFFTrain()

n_train=100

# create the task file contains the training, validation and testing dataset 
task=AFF_train.create_task(train_dataset=dataset, 
                            n_train = n_train ,
                            valid_dataset=dataset,
                            n_valid=50,
                            n_test=50,
                            lam = 1e-15,
                            uncertainty=False)

# start training the model based on the training dataset
trained_model = AFF_train.train(task,sig_candid_F = np.arange(10,20,10))

# predicted the force-field using the trained_model
prediction=AFF_train.predict(task = task, 
                             trained_model = trained_model,
                             R_test = task['R_test'][[0,1],:,:])

testing_force = task['F_test'][[0,1],:,:]

predicted_force = prediction['predicted_force']

MAE = np.mean(np.abs(  np.concatenate(testing_force)-np.concatenate(predicted_force)))

print("The MAE of testing dataset is "+str('{:f}'.format(MAE))+' \n')

#np.set_printoptions(suppress=True)