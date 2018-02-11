%% Set up dependencies
addpath('utility');
MatConvNetDir = 'D:\Repo\matconvnet'; % Change this to your MatConvNet path
VLFeatDir = 'D:\Libs\vlfeat-0.9.20'; % Change this to your VLFeat path
addpath(fullfile(VLFeatDir, 'toolbox')); vl_setup;
run(fullfile(MatConvNetDir, 'matlab', 'vl_setupnn.m'));

%% Settings
opts = struct;

opts.testDir = 'examples';
opts.modelDir = 'models';
opts.outDir = 'out';

opts.batchSize = 1;

%%

fileList = [dir(fullfile(opts.testDir, '*.jpg')); dir(fullfile(opts.testDir, '*.png'))];
fileList = arrayfun(@(x)(fullfile(opts.testDir, x.name)), fileList, 'UniformOutput', false);

faceDetPath = fullfile(opts.testDir, 'bbxDet.mat');
fprintf('Loading face detection (%s)\n', faceDetPath);
ldata = load(faceDetPath);
bbxDet = ldata.bbxDet;

if ~exist('model', 'var') || ~exist('net', 'var')
    modelPath = fullfile(opts.modelDir, 'pre-trained.mat');
    fprintf('Loading the model (%s)\n', opts.modelPath);
    model = load(opts.modelPath);
    net = dagnn.DagNN.loadobj(model.net);
    net.move('gpu');
    net.mode = 'test';
    model.net = net;
end

%%
fprintf('Refining the detection\n');
bbxRef = refinedet(fileList, bbxDet, model);

%%
fprintf('Classifying\n');
lmCluster = model.lmCluster;

optstt = struct;
optstt.outDir = fullfile(opts.outDir, 'classification');
optstt.bbxDet = bbxRef;
optstt.batchSize = opts.batchSize;

[lmPredict, infoClassify, labels, bbxMargin] = batchtestdiscrete(net, fileList, lmCluster, optstt);

%%
fprintf('Post-processing regression (trained with the same example sharing idea)\n');

initShape = lmPredict;
regModelDir = fullfile(opts.modelDir, 'ppreg');

optsreg = struct;
optsreg.outDir = opts.outDir;
optsreg.bbxDet = bbxRef;

[lmPredict, bbxMarginTest, info] = batchtestCReg(regModelDir, fileList, labels, initShape, optsreg);
