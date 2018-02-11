function [lmPredict, info, labels, bbxMargin, prob] = batchtestdiscrete(net, fileList, lmCluster, varargin)

opts = struct;
opts.color = [0 255 0];
opts.outDir = [];
opts.outPathList = [];
opts.radius = 3;
opts.bProb = false;
opts.batchSize = 1;
opts.bWriteImg = true;
opts.numThreads = 6;
opts.bbxDet = [];
opts.imageSize = [224 224];
opts.marginScale = 1.3;

opts = vl_argparse(opts, varargin);

n = length(fileList);

border = [];
averageImage = [];

bDet = ~isempty(opts.bbxDet);
if bDet
    bbxDet = opts.bbxDet;
    bbxMargin = cell(n, 1);
    imageSize = opts.imageSize;
    marginScale = opts.marginScale;
else
    bbxMargin = [];
end

labels = zeros(n, 1);
if opts.bProb
    outputIdxClass = net.getVarIndex('prob');
    outputIdx = outputIdxClass;
else
    outputIdx = net.getVarIndex('prediction');
end
[net.vars(outputIdx).precious] = deal(1);

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

if opts.bProb
    bProbInit = false;
else
    prob = [];
end

colors = opts.color;

lmPredict = cell(n, 1);
tCNN = zeros(n, 1);
tPerFrame = zeros(n, 1);

if ~isempty(opts.outDir)
    mkdir2(opts.outDir);
end

for b = 1:opts.batchSize:n
    thisBatchSize = min(opts.batchSize, n - b + 1);
    batchEnd = b + thisBatchSize - 1;
    batch = b: batchEnd;
    if thisBatchSize == 1
        fprintf('Processing image %d/%d (%s)\n', b, n, fileList{b});
    else
        fprintf('Processing image %d-%d/%d (%s)\n', b, batchEnd, n, fileList{b});
    end
    tStart1 = tic;
    
    imgCells = vl_imreadjpeg(fileList(batch), 'numThreads', opts.numThreads);
    if bDet
        for i = 1:thisBatchSize
            idx = batch(i);
            if isempty(bbxDet{idx})
                imgCells{i} = zeros([imageSize 3], 'like', imgCells{i});
                bbxMargin{idx} = [1 1 imageSize];
            else
                [~, bbxMargin{idx}, ~, imgCells{i}] = cropface(imgCells{i}, bbxDet{idx}, imageSize, marginScale);
            end
        end
    end
    inImgs = cat(4, imgCells{:});
    if ~isempty(border)
        inImgs = inImgs(border(1)/2+1:end-border(1)/2,border(2)/2+1:end-border(2)/2, :, :);
    end
    tStart2 = tic;

    img = gpuArray(inImgs);
    if ~isempty(averageImage)
        img = bsxfun(@minus, img, averageImage);
    end
    net.eval({'input', img});
    if opts.bProb
        probBatch = squeeze(gather(net.vars(outputIdx).value)); % [C N]
        if ~bProbInit
            nClass = size(probBatch, 1);
            prob = zeros(nClass, n);
            bProbInit = true;
        end
        prob(:, batch) = probBatch;
        [probMax, labels(batch)] = max(probBatch, [], 1);
    else
        prediction = squeeze(gather(net.vars(outputIdx).value));
        [~, labels(batch)] = max(prediction, [], 1);
    end

    tElapsed2 = toc(tStart2);
    tCNN(batch) = tElapsed2/thisBatchSize;
    
    for i = 1:thisBatchSize
        idx = batch(i);
        lmPredict{idx} = lmCluster(:, :, labels(idx));
        if opts.bWriteImg
            outImg = insertShape(uint8(imgCells{i}), 'FilledCircle', [lmPredict{idx} opts.radius*ones(size(lmPredict{idx}, 1), 1)], 'Color', colors, 'Opacity', 0.8);

            if isempty(opts.outPathList)
                [~, fileName, fileExt] = fileparts(fileList{idx});
                outPath = fullfile(opts.outDir, [fileName fileExt]);
            else
                outPath = opts.outPathList{idx};
                mkdir2(fileparts(outPath));
            end
            imwrite(outImg, outPath);
        end
    end
    
    tElapsed1 = toc(tStart1);
    tPerFrame(batch) = tElapsed1/thisBatchSize;
end

info = struct;
info.tCNN = tCNN;
info.tPerFrame = tPerFrame;
    
end