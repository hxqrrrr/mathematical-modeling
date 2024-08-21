function [ Row ] = rowDetect( id )
%rowDetect is to get each line beginning pixal.
%   Row is a vector of each line beginning pixal.
%   id is the picture number.
    example=imread(id);
    after=im2bw(example,graythresh(example));
    rowLength=length(after);
    flag=1;
    count=1;
    Row=zeros;
    j=1;
    while(j~=rowLength)
        number=length(find(after(j,:)==0));
        if number~=0 
            if flag==1
                Row(count)=j;
                count=count+1;
                flag=0;
                j=j+39;
            end
        end

        if (flag==0 && number==0)
            flag=1;
        end
        j=j+1;
    end

end

