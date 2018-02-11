function [bbxFace, bbxMargin, transAnnot, outImg] = cropface(inImg, annot, varargin)

if nargin < 3
    outSize = [256 256];
else
    outSize = varargin{1};
    if nargin < 4
        marginScale = 1.5;
    else
        marginScale = varargin{2};
        if nargin < 5
            fmt = 1;
        else
            fmt = varargin{3};
        end
    end
end

% if size(annot, 3) > 1, then the first one is the anchor, and we also need to
% transfrom the rest as well (currently only supports landmarks)
bTransformAdditional = size(annot, 3) > 1;
if bTransformAdditional
    annotAll = annot;
    annot = annot(:, :, 1);
end

if size(annot, 2) == 2
    bLandmarks = true;
    xgt = annot(:, 1);
    ygt = annot(:, 2);
    bbxFace = [min(xgt) min(ygt) max(xgt) max(ygt)];
else
    % used during refined detection where the boxes are real numbers
    bLandmarks = false;
    bbxFace = [annot(1:2), annot(1) + annot(3), annot(2) + annot(4)];
    transAnnot = [];
end
cFace = [(bbxFace(1) + bbxFace(3))/2, (bbxFace(2) + bbxFace(4))/2];
if fmt == 1
    rFace = (max(bbxFace(3) - bbxFace(1), bbxFace(4) - bbxFace(2)) + 1)/2;
else
    rFace = ((bbxFace(3) - bbxFace(1)) + (bbxFace(4) - bbxFace(2)))/4;
end

rMargin = rFace * marginScale;
bbxMargin(1) = round(cFace(1) - rMargin);
bbxMargin(2) = round(cFace(2) - rMargin);
lMargin = ceil(2*rMargin);
bbxMargin(3) = bbxMargin(1) + lMargin - 1;
bbxMargin(4) = bbxMargin(2) + lMargin - 1;

if bTransformAdditional
    transAnnot = annotAll;
    transAnnot(:, 1, :) = transAnnot(:, 1, :) - bbxMargin(1);
    transAnnot(:, 2, :) = transAnnot(:, 2, :) - bbxMargin(2);
else
    if bLandmarks
        transAnnot = annot;
        transAnnot(:, 1) = transAnnot(:, 1) - bbxMargin(1);
        transAnnot(:, 2) = transAnnot(:, 2) - bbxMargin(2);
    end
end

inSize = size(inImg);
if length(inSize) < 3
    % grayscale
    inImg = inImg(:, :, [1 1 1]);
end
outImg = zeros(lMargin, lMargin, 3, 'like', inImg);

if bbxMargin(1) <= 0
    inLeft = 1;
    outLeft = -bbxMargin(1) + 2;
else
    inLeft = bbxMargin(1);
    outLeft = 1;
end

if bbxMargin(3) > inSize(2)
    inRight = inSize(2);
    outRight = inSize(2) - bbxMargin(1) + 1;
else
    inRight = bbxMargin(3);
    outRight = lMargin;
end

if bbxMargin(2) <= 0
    inTop = 1;
    outTop = -bbxMargin(2) + 2;
else
    inTop = bbxMargin(2);
    outTop = 1;
end

if bbxMargin(4) > inSize(1)
    inBottom = inSize(1);
    outBottom = inSize(1) - bbxMargin(2) + 1;
else
    inBottom = bbxMargin(4);
    outBottom = lMargin;
end
outImg(outTop:outBottom, outLeft:outRight, :) = inImg(inTop:inBottom, inLeft:inRight, :);

outImg = imresize(outImg, outSize);

if bTransformAdditional
    transAnnot(:, 1, :) = transAnnot(:, 1, :) * (outSize(2)/lMargin);
    transAnnot(:, 2, :) = transAnnot(:, 2, :) * (outSize(1)/lMargin);
else
    if bLandmarks
        transAnnot(:, 1) = transAnnot(:, 1) * (outSize(2)/lMargin);
        transAnnot(:, 2) = transAnnot(:, 2) * (outSize(1)/lMargin);
    end
end

end