function []=feature_encoding(arg1, arg2, arg3, arg4, arg5)
    % arg1 - path to CSV file with centroids
    % arg2 - type of encoding (bow or vlad)
    % arg3 - path to vlfeat/ directory
    % arg4 - path to directory with all .key files
    % arg5 - path to output file 
    addpath(arg3);
    run([arg3 '/toolbox/vl_setup'])
    centroids = dlmread(arg1, ',');
    kdtree = vl_kdtreebuild(transpose(centroids));
    [status,files] = system(['ls ' arg4]);
    files = strsplit(files);
    files = files(1:(length(files)-1));

    if strcmp(arg2, 'vlad')
        vlad_enc = zeros(0,prod(size(centroids))+1);
        for i = 1:length(files)
            myfile = char(strcat(arg4, '/', files(i)));
            tempdat = dlmread(myfile, '\t', 6, 0);
            tempdat = tempdat(:,18:81);
            nn = vl_kdtreequery(kdtree, transpose(centroids), transpose(tempdat));
            assignments = zeros(size(centroids,1), length(tempdat));
            assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
            enc = vl_vlad(transpose(tempdat), transpose(centroids), assignments);
            id = str2num(char(strrep(files(i), '.key', '')));
            vlad_enc = [vlad_enc; [transpose(enc), id]];
        end
        dlmwrite(arg5, vlad_enc, ',');
    elseif strcmp(arg2, 'bow')
        bow_enc = zeros(0,size(centroids,1)+1);
        for i = 1:length(files)
            myfile = char(strcat(arg4, '/', files(i)));
            tempdat = dlmread(myfile, '\t', 6, 0);            
            tempdat = tempdat(:,18:81);
            nn = vl_kdtreequery(kdtree, transpose(centroids), transpose(tempdat));
            tempbow = hist(nn, 1:size(centroids,1));
            tempbow = tempbow / sum(tempbow);
            id = str2num(char(strrep(files(i), '.key', '')));
            tempbow = [tempbow, id];
            bow_enc = [bow_enc; tempbow];
        end
        dlmwrite(arg5, bow_enc, ',');
    end
    disp('DONE!');
end
