function desc = hogwrapper(im, pos, lmsize)

rect =  [pos(1) - (lmsize-1)/2, ...
         pos(2) - (lmsize-1)/2, ...
         lmsize - 1, lmsize - 1];
     
cropim = imcrop(im,rect);
if isempty(cropim)
    cropim = zeros([lmsize lmsize size(im, 3)], 'like', im);
end
   
if size(cropim,1) ~= lmsize || size(cropim,2) ~= lmsize
     cropim = imresize(cropim,[lmsize lmsize]);
end

cellSize = 32 ;
%tmp = vl_hog(single(cropim), cellSize, 'verbose');
tmp = vl_hog(cropim, cellSize);

%desc = feat_normalize(tmp(:));
desc = tmp(:);
     
end