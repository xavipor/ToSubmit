clear;clc;
addpath('./NIfTI_20140122/')

mode = 'test';
img_data_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/data/TheirSizes/data/'

%img_data_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/data/newImages/NewDataZ1_4/DataLessZ/';
%img_data_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/data/DataResampledV2/';
%img_data_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/data/raw_data/';
save_datasets_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/result/mat_data/';
if ~exist(save_datasets_path)
    mkdir(save_datasets_path);
end

x = 596;%536
y = 596;%596
z = 170;

%x = 512
%y = 512
%z= 150 

files = dir(img_data_path);
files(1:2)=[]%To delete the first to elements that are junk
num = length(files);

fprintf('Extracting %s dataset ... \n',mode);
for jj = 1:num
    data_volume = zeros(x,y,z);
    display(prod(size(data_volume)));
    fprintf('Loading No.%d %s subject (total %d).\n', jj,mode,num);
    nii = load_untouch_nii([img_data_path files(jj).name]);
    [xrange yrange zrange] = size(nii.img);
    normalize = (nii.img - min(nii.img(:)))./(max(nii.img(:)) - min(nii.img(:)));
    aux = normalize(:);
    aux2 = aux - mean(aux);
    aux3 = reshape(aux2,[xrange,yrange,zrange]); 
    data_volume(1:xrange,1:yrange,1:zrange) = aux3(:,:,:);
    data_volume = data_volume(1:end,1:end,1:end-2);
    
    
    data = reshape(data_volume,[1 prod(size(data_volume))]);
    save([save_datasets_path num2str(jj) '_' mode '.mat'],'data','-v7.3');  
    clear nii data data_volume
end

exit




    
    
