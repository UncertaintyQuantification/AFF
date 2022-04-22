# AFF
Software package for atomized force fields emulator (Python) 

Description: This package is the application of AFF method. It allows the efficient molecular potential energy surface and force fields emulation. 

Reference: Efficient force field and energy emulation through partition of permutationally equivalent atoms. The Journal of Chemical Physics. (https://aip.scitation.org/doi/10.1063/5.0088017) 

Requirement: Python 3.7+, NumPy (>=1.19) ,SciPy (>=1.1)

Example: Query a force field
```
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

# train the model based on the training dataset
trained_model = AFF_train.train(task,sig_candid_F = np.arange(10,20,10))

# predict the force-field using the trained_model
prediction=AFF_train.predict(task = task, 
                             trained_model = trained_model,
                             R_test = task['R_test'][[0,1],:,:])
# force field prediction
predicted_force = prediction['predicted_force']

```
