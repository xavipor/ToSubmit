function get_score_map_cand(result_path,dimx,dimy,dimz,threshold_score_mask,i)
    %threshold_score_mask: the threshold to obtain the candidates from score map, ranging between [0 1]
  
    score_map_path = [result_path 'score_map/'];   %geenrated by first net
    aux_path = strcat('score_map_cands/treshold',string(i*10),'/'); 
    cand_path = strcat(result_path,aux_path) % the path of candidates from score map    
    cand_path = char(cand_path);
    M_layer = 1;
    
    if ~isdir(cand_path)
        mkdir(cand_path);
    end
    files = dir(score_map_path);
    files(1:2) = [];%This is just because the first two lineas are junk
    for jj = 1:length(files)      
        fprintf('Loading No.%d testing subject (total %d).\n', jj,length(files));        
        load([score_map_path num2str(jj) '_score_mask.mat']);
        sz_sp = size(score_mask);%[2 249 249 70]
        %prueba poniendo un 1 en lo que queremos coger si no, no tiene puto sentido
%        score_map = reshape(score_mask(2,:,:,:),sz_sp(2:end));%We take justhe postive layer I suppose
        score_map = reshape(score_mask(2,:,:,:),sz_sp(2:end));%We take justhe postive layer I suppose
        filtered_score_map = peak_score_map(score_map); %We just keep the values of the pixels that happens to be equal to a maximum after applying a filter  
%        [mask center_score_map] = get_proposal_from_score_map_all_count(score_map,threshold_score_mask); %Based on the use of in2sub get the  position of pixels with values bigger than the treshold
        [mask center_score_map] = get_proposal_from_score_map_all_count(filtered_score_map,threshold_score_mask); %Based on the use of in2sub get the  position of pixels with values bigger than the treshold
%        center = [2*M_layer*(center_score_map(:,2)-1)+dimx/2,2*M_layer*(center_score_map(:,1)-1)+dimy/2,2*M_layer*(center_score_map(:,3)-1)+dimz/2];
        center = [2*M_layer*(center_score_map(:,1)-1)+dimx/2,2*M_layer*(center_score_map(:,2)-1)+dimy/2,2*M_layer*(center_score_map(:,3)-1)+dimz/2];
        %mask is just a logial array of size 249 249 70 with ones in the
        %places where the valie of the pixels is bigger than the treshold
        %center socre map 317x3 has the coordinates of those pixels at the end of
        %the convolutional net. Center are the same values but in the
        %original image..
%        center = permute(center,[2,1,3]);
        path_filtered = strcat('filteredJavi/',string(jj),'.mat');
        path_filtered_total = char(strcat(result_path,path_filtered));
        save(path_filtered_total,'filtered_score_map');
        save([cand_path num2str(jj) '_cand.mat'],'center');
    end       
end










