function [ distance ] = dis( data )
%dis is to calculate the distance of one picture.
%   distance is the result. 

    [~,N]=size(data);
    for i=1:N
        
        if ~isempty(find(data(:,i)==0))
            distance(1,1)=i-1;
            break;
        end
    end
    
    for j=N:-1:1
      
        if ~isempty(find(data(:,j)==0))
            distance(1,2)=N-j;
            break;
        end
    end
    
end

