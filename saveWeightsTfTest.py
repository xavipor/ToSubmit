
"""
Get the values saved by the authors of the paper and change the order of the dimensions due 
to de requirements of the Convolutional Layers of TensorFlow
"""

import numpy as np
import cPickle
import pdb
"""
model_path ='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/model/fine_tuned_params.pkl'
f_param = open(model_path,'r')
params = cPickle.load(f_param)
W0,b0,W1,b1,W2,b2,W3,b3,W4,b4 = [param for param in params]

W_L0 = (1-0.2)*W0.get_value()#([64,  3,  1,  5,  5])#
W_L0 = W_L0.transpose(1,3,4,2,0)
b_L0 = b0.get_value()#[64] 

W_L1 = (1-0.3)*W1.get_value()#([64,  3, 64,  3,  3])
W_L1 = W_L1.transpose(1,3,4,2,0)
b_L1 = b1.get_value()#[64]

W_L2 = (1-0.3)*W2.get_value()#([64,  1, 64,  3,  3])
W_L2 = W_L2.transpose(1,3,4,2,0)
b_L2 = b2.get_value()#[64]

W_L3 =(1-0.3)*W3.get_value()
W_L3 = W_L3.transpose(1,3,4,2,0)
b_L3 =b3.get_value()

W_L4 =(1-0.3)*W4.get_value()
W_L4 = W_L4.transpose(1,3,4,2,0)
b_L4 = b4.get_value()




path='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/PesosPruebaTf/'

np.save(path+'W_L0',W_L0)
np.save(path+'b_L0',b_L0)

np.save(path+'W_L1',W_L1)
np.save(path+'b_L1',b_L1)

np.save(path+'W_L2',W_L2)
np.save(path+'b_L2',b_L2)

np.save(path+'W_L3',W_L3)
np.save(path+'b_L3',b_L3)

np.save(path+'W_L4',W_L4)
np.save(path+'b_L4',b_L4)

"""





model_path ='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/model/fine_tuned_params_step2.pkl'
f_param = open(model_path,'r')
params = cPickle.load(f_param)

params_L0, params_L1, params_L2, params_L3, params_L4 = [param for param in params]

pdb.set_trace()
W_L0 = params_L0[0].get_value()
W_L0 = W_L0.transpose(1,3,4,2,0)
b_L0 = params_L0[1].get_value()

W_L1 = params_L1[0].get_value()
W_L1 = W_L1.transpose(1,3,4,2,0)
b_L1 = params_L1[1].get_value()

W_L2 = params_L2[0].get_value()
b_L2 = params_L2[1].get_value()

W_L3 = params_L3[0].get_value()
b_L3 = params_L3[1].get_value()

W_L4 = params_L4[0].get_value()
b_L4 = params_L4[1].get_value()




path='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/PesosPruebaTf2/'

np.save(path+'W_L0',W_L0)
np.save(path+'b_L0',b_L0)

np.save(path+'W_L1',W_L1)
np.save(path+'b_L1',b_L1)

np.save(path+'W_L2',W_L2)
np.save(path+'b_L2',b_L2)

np.save(path+'W_L3',W_L3)
np.save(path+'b_L3',b_L3)

np.save(path+'W_L4',W_L4)
np.save(path+'b_L4',b_L4)
