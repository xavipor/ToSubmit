import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pdb
def load_from_file(filename):
    """ Load object from file
    """
    object = []
    f = open(filename + '.pckl', 'rb')
    object = pickle.load(f)
    f.close()
    return object
path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModel/31082018_Z14_NewDistribution_AllLayers_Reg_DROPOUT/'
path2 = '_0.001_5000_0.03'
errTraining = load_from_file(path + 'ErrorT' + path2)
errVal = load_from_file(path + 'ErrorV' + path2)
accTraining = load_from_file(path + 'AccT' + path2)
accValidation = load_from_file(path + 'AccV' + path2)

print("Training Error ----->  " , errTraining[12])
print("Validation Error ----->  ", errVal[12])
print("Training Accuracy ----->  ", accTraining[12])
print("Validation Accuracy ----->  ", accValidation[12])

fig_1 = plt.figure(figsize=(50,100))
ax_1 = fig_1.add_subplot(1, 2, 1)
ax_2 = fig_1.add_subplot(1, 2, 2)

l = sorted(errTraining.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Training LR = 0.001")

l = sorted(errVal.items())
x,y = zip(*l)
ax_1.plot(x,y,label="Error on Validation LR = 0.001")

l = sorted(accTraining.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Training LR = 0.001")

l = sorted(accValidation.items())
x,y = zip(*l)
ax_2.plot(x,y,label="Accuracy on Validation LR = 0.001")

ax_1.set_title("Error on validation and training set with Adam 2 FC Layers No Reg")
ax_2.set_title("Accuracy on validation and training set with Adam 2 FC Layers No Reg")
ax_1.set_ylabel ("value")
ax_1.set_xlabel ("epochs")
ax_1.legend()
ax_2.set_ylabel ("value")
ax_2.set_xlabel ("epochs")
ax_2.legend()
pdb.set_trace()
plt.show()
