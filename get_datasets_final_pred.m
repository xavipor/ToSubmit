function get_datasets_final_pred(result_path,xdim,ydim,zdim)
    % This script is to extract cubic candidates to wrap into the final
    % prediction model
%    img_data_path ='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/data/DataResampledV2/' ;
%    img_data_path ='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/data/raw_data/'
    img_data_path ='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/data/TheirSizes/data/'
%    img_data_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/data/newImages/NewDataZ1_4/DataLessZ/';
    files = dir(img_data_path);
    files(1:2) = [];
    score_map_cand_path = [result_path 'score_map_cands2/'];
    datasets_path = [result_path 'test_set_cand/']; 
    if ~exist(datasets_path)
        mkdir(datasets_path);
    end   
    % Start to extract the test candidates cubics %%%%%%%%%%%%%%%%%%%%%%
    fprintf('Extracting testing dataset ...\n');
    tic;
    for jj = 1:length(files)
        counter = 0;
        test_set_x = zeros([10000 xdim*ydim*zdim]);  %To reserve memory... funny thing of doing this. 
        fprintf('Loading No.%d test subject (total %d).\n', jj, length(files));
        nii = load_untouch_nii([img_data_path sprintf('%02d',jj) '.nii.gz']);
        [xrange yrange zrange] = size(nii.img);
        normalize = (nii.img - min(nii.img(:)))./(max(nii.img(:)) - min(nii.img(:)));
	aux = normalize(:);
	aux2 = aux - mean(aux);
        nii_img = reshape(aux2,[xrange,yrange,zrange]);
        load([score_map_cand_path num2str(jj) '_cand.mat']);
        [b_x b_y b_z] = size(nii.img);
        for i = 1:size(center,1)
            counter = counter + 1;
            pp1 = center(i,1);
            pp2 = center(i,2);
            pp3 = center(i,3);  
            %Basically we eliminite as candidates pixels that cant be
            %centers of the patches because they are not going to be able
            %to support the patch without end up out of the image. 
            if (pp1<xdim/2 || pp1>b_x-xdim/2 || pp2<ydim/2 || pp2>b_y-ydim/2 || pp3<zdim/2 || pp3>b_z-zdim/2)
                patch = zeros(xdim,ydim,zdim);
            else
                patch = nii_img(pp1-xdim/2+1:pp1+xdim/2,pp2-ydim/2+1:pp2+ydim/2,pp3-zdim/2+1:pp3+zdim/2);
            end
            test_set_x(counter,:) = reshape(patch, xdim*ydim*zdim, []);       
            clear patch
        end
        test_set_x(counter+1:end,:)=[];%To discard the unused part of the big tensor
        save([datasets_path num2str(jj) '_patches.mat'],'test_set_x','-v7.3');
    end
    toc;
end









