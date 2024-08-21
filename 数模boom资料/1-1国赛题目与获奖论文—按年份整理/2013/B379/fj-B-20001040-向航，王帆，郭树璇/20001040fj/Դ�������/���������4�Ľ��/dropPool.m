function [ pool ] = dropPool( picturePool,number )
%dropPool is to drop number from picturePool.
%   picturePool is the pool of picture.
%   number is the drop number.
        
        if isempty(picturePool)
           pool=[];
           return
        end

        count=1;
    for i=1:length(picturePool)
        if length(picturePool)==1
            pool=[];
        end
        if picturePool(i)==number
            continue;
        end
        pool(count)=picturePool(i);
        count=count+1;
    end
end

