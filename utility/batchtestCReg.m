function [lmPredict, bbxMargin, info] = batchtestCReg(modelDir, fileList, labels, initShape, varargin)

opts = struct;
opts.radius = 3;
opts.outDir = [];
opts.bWriteImg = true;
opts.bbxDet = [];
opts.imageSize = [224 224];
opts.marginScale = 1.3;
opts.bParentInit = true;
opts.flipIndicator = [];
opts.regressorSwitch = [];
opts.bMinVis = true;

opts = vl_argparse(opts, varargin);

n = length(fileList);

if ~isempty(opts.outDir)
    mkdir2(opts.outDir);
end

% label mapping to the parent level
ldata = load(fullfile(modelDir, 'imdb.mat'));
imdb = ldata.imdb;
labelMapping = [imdb.labels(imdb.train); imdb.labelsFlip(imdb.train)];  
labels = labelMapping(labels);
if opts.bParentInit
    ldata = load(fullfile(modelDir, 'lmCluster.mat'));
    lmCluster = ldata.lmCluster;
    for i = 1:n
        initShape{i} = lmCluster(:, :, labels(i));
    end
end

[labels, sIdx] = sort(labels);
initShape = initShape(sIdx);
fileList = fileList(sIdx);

imageSize = opts.imageSize;
marginScale = opts.marginScale;
flipIndicator = opts.flipIndicator;
regressorSwitch = opts.regressorSwitch;
if ~isempty(flipIndicator)
    flipIndicator = flipIndicator(sIdx);
end
if ~isempty(regressorSwitch)
    regressorSwitch = regressorSwitch(sIdx);
end

bDet = ~isempty(opts.bbxDet);
if bDet
    bbxDet = opts.bbxDet(sIdx);
    bbxMargin = cell(n, 1);
else
    bbxMargin = [];
end
nFeat = 124;
dSize = 64;

nLandmarks = size(initShape{1}, 1);
lmPredict = cell(n, 1);
tPerFrame = zeros(n, 1);
bModelExist = false(n, 1);
X = zeros(nLandmarks, nFeat);

preLabel = 0;
for i = 1:n
    fprintf('Processing image %d/%d (%s)\n', i, n, fileList{i});
    tStart = tic;
    if isempty(regressorSwitch) || regressorSwitch(i)
        if preLabel == labels(i)
            bModelExist(i) = bModelExist(i-1);
        else
            preLabel = labels(i);
            modelPath = fullfile(modelDir, num2str(preLabel, '%06u.mat'));
            if exist(modelPath, 'file')
                bModelExist(i) = true;
                ldata = load(modelPath);
                W = ldata.W;
                if isempty(W{1})
                    bModelExist(i) = false;
                    warning('Empty model found. Might be errors in training.');
                end
            end
        end
    end

    img = imread(fileList{i});
    if bDet
        if isempty(bbxDet{i})
            img = zeros([imageSize 3], 'uint8');
            bbxMargin{i} = [1 1 imageSize];
        else
            [~, bbxMargin{i}, ~, img] = cropface(img, bbxDet{i}, imageSize, marginScale);
        end
    end
    
    if bModelExist(i)
        inImg = img;
        if size(inImg,3) == 3
            inImg = rgb2gray(inImg);
        end
        if ~isempty(flipIndicator)&& flipIndicator(i)
            inImg = fliplr(inImg);
        end
        inImg = imresize(inImg, [400 400]);
        inImg = im2single(inImg);
        curShape = initShape{i};
        for c = 1:length(W)
            lm = curShape*(400/224);
            for j = 1:nLandmarks
                X(j, :) = hogwrapper(inImg, lm(j, :), dSize);
            end
            Y = W{c}'*X(:);
            Y = reshape(Y, nLandmarks, 2);
            curShape = curShape + Y;
        end
        lmPredict{i} = curShape;
    else
        lmPredict{i} = initShape{i};
    end
    
    if opts.bWriteImg
        outImg = img;
        if opts.bMinVis
            outImg = insertShape(outImg, 'FilledCircle', [lmPredict{i} opts.radius*ones(nLandmarks, 1)], 'Color', [0 255 0], 'Opacity', 0.8);
        else
            outImg = insertShape(outImg, 'FilledCircle', [initShape{i} opts.radius*ones(nLandmarks, 1)], 'Color', [0 255 0], 'Opacity', 0.8);
            if bModelExist(i)
                outImg = insertShape(outImg, 'FilledCircle', [lmPredict{i} opts.radius*ones(nLandmarks, 1)], 'Color', [255 0 255], 'Opacity', 0.8);
            end
            outImg = insertText(outImg, opts.textPos, labels(i), 'BoxColor', 'cyan');
        end
        
        [~, fileName, fileExt] = fileparts(fileList{i});
        outPath = fullfile(opts.outDir, [fileName fileExt]);
        imwrite(outImg, outPath);
    end
    
    tPerFrame(i) = toc(tStart);
end

lmPredict(sIdx) = lmPredict;
if ~isempty(bbxMargin)
    bbxMargin(sIdx) = bbxMargin;
end
bModelExist(sIdx) = bModelExist;

info = struct;
info.tPerFrame = tPerFrame;

end