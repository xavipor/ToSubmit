import tensorflow as tf
import numpy as np

pathToSave = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/WeightsTrainedSN/'

# Add ops to save and restore all the variables.
#saver = tf.train.import_meta_graph("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/AllNetGDZ14/my_test_model_GD26_test_z14_0.001_5000_0.003-4200.meta")
saver = tf.train.import_meta_graph("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModel/23082018_SecondNet_TS_No_Reg_3FC_Layers_SecondChance/my_test_model_0.001_5000_0-17.meta")

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess,("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModel/23082018_SecondNet_TS_No_Reg_3FC_Layers_SecondChance/my_test_model_0.001_5000_0-17"))
  print("Model restored.")
  # Check the values of the variables and save them
  for i in range(5):
	Wt = sess.graph.get_tensor_by_name("W"+str(i)+":0")
	bt = sess.graph.get_tensor_by_name("b"+str(i)+":0")
	W = Wt.eval() #To get a numpy array from the tensor
	print(W.shape)
	b = bt.eval() #To get a numpy array from the tensor
	np.save(pathToSave+"W"+str(i),W)
	np.save(pathToSave+"b"+str(i),b)
  sess.close()

print("All weights and biases saved" )
