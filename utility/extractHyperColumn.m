function ft = extractHyperColumn(net, fileList, varargin)

opts = struct;
opts.verbosity = 1;
% opts.featList = {'res5cx'};
opts.featList = {'res4fx', 'res5cx'};
% opts.featList = {'conv1xxx', 'res2cx', 'res3dx', 'res4fx', 'pool5'}; % res5cx
opts.batchSize = 1;
opts.numThreads = 8;
opts.imageSize = [224 224];
opts.marginScale = 1.3;
opts.scale = [];
opts.scale = 1/16;
opts.bbxDet = [];
opts.flipIndicator = [];
opts.ftDir = [];

opts = vl_argparse(opts, varargin);

nFeat = length(opts.featList);
featIdx = net.getVarIndex(opts.featList);
[net.vars(featIdx).precious] = deal(1);

averageImage = [];
border = [];
bNormalization = any(strcmp(properties(net), 'meta')) && isfield(net.meta, 'normalization');
% Note object is not a struct, cannot be replaced with isfield(net, 'meta')!
if bNormalization
    if isfield(net.meta.normalization, 'border')
        border = net.meta.normalization.border;
        if isequal(border, [0 0])
            border = [];
        end
    end
    if isfield(net.meta.normalization, 'averageImage')
        averageImage = net.meta.normalization.averageImage;
        if numel(averageImage) == 3
            averageImage = reshape(averageImage, 1, 1, 3);
        end
    end
end
    
n = length(fileList);

bDet = ~isempty(opts.bbxDet);
if bDet
    bbxDet = opts.bbxDet;
end
imageSize = opts.imageSize;
marginScale = opts.marginScale;
if isempty(opts.scale)
    resizeSize = [];
elseif opts.scale == 0
    resizeSize = 0;
else
    resizeSize = round(opts.scale*imageSize);
end

flipIndicator = opts.flipIndicator;

bSaveToFile = ~isempty(opts.ftDir);
if bSaveToFile
    ftDir = opts.ftDir;
else
    ft = cell(n, 1);
end

for b = 1:opts.batchSize:n
    thisBatchSize = min(opts.batchSize, n - b + 1);
    batchEnd = b + thisBatchSize - 1;
    batch = b: batchEnd;
    if opts.verbosity >= 1
        if thisBatchSize == 1
            fprintf('Processing image %d/%d (%s)\n', b, n, fileList{b});
        else
            fprintf('Processing image %d-%d/%d (%s)\n', b, batchEnd, n, fileList{b});
        end
    end
    
    imgCells = vl_imreadjpeg(fileList(batch), 'numThreads', opts.numThreads);
    if bDet
        for i = 1:thisBatchSize
            idx = batch(i);
            assert(~isempty(bbxDet{idx}));
            [~, ~, ~, imgCells{i}] = cropface(imgCells{i}, bbxDet{idx}, imageSize, marginScale);
        end
    end
    if ~isempty(flipIndicator)
        for i = 1:thisBatchSize
            if flipIndicator(batch(i))
                imgCells{i} = fliplr(imgCells{i});
            end
        end
    end
    inImgs = cat(4, imgCells{:});
    if ~isempty(border)
        inImgs = inImgs(border(1)/2+1:end-border(1)/2,border(2)/2+1:end-border(2)/2, :, :);
    end
    img = gpuArray(inImgs);
    if ~isempty(averageImage)
        img = bsxfun(@minus, img, averageImage);
    end
    %             [net.vars(:).precious] = deal(1); % used for debugging
    net.eval({'input', img});
    vals = cell(nFeat, 1);
    for j = 1:nFeat
        vals{j} = net.vars(featIdx(j)).value;
        if ~isempty(resizeSize)
            if isequal(resizeSize, 0)
                % pick the center pixel
                h = size(vals{j}, 1);
                w = size(vals{j}, 2);
                vals{j} = vals{j}(round(h/2), round(w/2), :, :); 
            else
                vals{j} = imresize(vals{j}, resizeSize);
            end
        end
    end
    ftMat = gather(cat(3, vals{:}));
    if bSaveToFile
        for i = 1:thisBatchSize
            idx = batch(i);
            ft = ftMat(:, :, :, i);
            save(fullfile(ftDir, num2str(idx, '%06u.mat')), 'ft');
        end
    else
        for i = 1:thisBatchSize
            idx = batch(i);
            ft{idx} = ftMat(:, :, :, i);
        end
    end
end

if bSaveToFile
    ft = [];
end

end