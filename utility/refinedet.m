function bbxRef = refinedet(fileList, bbxDet, model)

n = length(fileList);

bbxRef = cell(n, 1);

net = model.net;
refModel = model.refModel;

for i = 1:n
    fprintf('Processing image %u/%u (%s)\n', i, n, fileList{i});
    bbxD = bbxDet{i};
    if ~isempty(bbxD)
        % transform the box to a square one
        lDet = max(bbxD(3:4));
        cDet = bbxD(1:2) + bbxD(3:4)/2;
        % bias correction for MTCNNv2
        lBC = lDet*refModel.meanSGtDet;
        bbxBC = [cDet + refModel.meanDGtDet*lDet - lBC/2, lBC, lBC];
        cBC = bbxBC(1:2) + bbxBC(3:4)/2;
        
        optsft = struct;
        optsft.bbxDet = {bbxBC};
        optsft.verbosity = 0;
        X = extractHyperColumn(net, fileList(i), optsft);
        X = X{1}(:);
        Yx = predict(refModel.x, X, 'ObservationsIn', 'columns');
        Yy = predict(refModel.y, X, 'ObservationsIn', 'columns');
        Ys = predict(refModel.s, X, 'ObservationsIn', 'columns');
                
        lRef = lBC*exp(Ys);
        bbxR = [cBC + [Yx Yy]*lBC - lRef/2, lRef, lRef];
        
        bbxRef{i} = bbxR;
    end
end

end