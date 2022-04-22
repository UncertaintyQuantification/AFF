#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 14:21:08 2022

@author: hao li
"""
import numpy as np
from utils import aff

# load the dataset contains the geometry information and the force-fields

dataset=np.load('./dataset/aspirin_ccsd-train.npz')


AFF_train=aff.AFFTrain()

n_train=100

# create the task file contains the training, validation and testing dataset 
task=AFF_train.create_task(train_dataset=dataset, 
                            n_train = n_train ,
                            valid_dataset=dataset,
                            n_valid=50,
                            n_test=50,
                            lam = 1e-15)

# start training the model based on the training dataset
trained_model = AFF_train.train(task,sig_candid_F = np.arange(10,100,10))

# predicted the force-field using the trained_model
prediction=AFF_train.predict(task = task, trained_model = trained_model)

#testing_force = task['F_test']
#predicted_force = prediction['predicted_force']
#MAE = np.mean(np.abs(  np.concatenate(testing_force)-np.concatenate(predicted_force)))

#print(MAE)