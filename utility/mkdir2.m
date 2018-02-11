function pathName = mkdir2(pathName)
% make a dir if the directory does not exist
% and returns the directory name

if ~exist(pathName, 'dir')
    mkdir(pathName);
end

end

