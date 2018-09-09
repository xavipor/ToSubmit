clear;clc;
addpath('/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/post_processing');
addpath('/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/NIfTI_20140122/')
addpath('/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/MinMaxFilterFolder');
result_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/result/';
%result_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/data/TheirSizes/myTest/Training/';
%% get the score map candidates
dimx = 16;
dimy = 16;
dimz = 10;
T = [0:0.05:0.95];
T = cat(2,T,[0.96:0.0001:1])
for i=1:1:length(T)
	threshold_score_mask = T(i);
	display(result_path)
	get_score_map_cand(result_path,dimx,dimy,dimz,threshold_score_mask,i);
end
exit
%% get stage-2 patches
dimx = 20;
dimy = 20;
dimz = 16;
get_datasets_final_pred(result_path,dimx,dimy,dimz);
% get_datasets_final_pred_dundee(result_path,dimx,dimy,dimz)
