function [Edge,row] = match( Length )
%match is to get the edge of picture and get the each line beginning pixal.
%   Length is the number of dataset.
%   Edge is the picture edge.
%   row is the beginning of each line.
    
    n=Length;
    for i=1:n
        if i<=10
            img=strcat(strcat('00',int2str(i-1)),'.bmp');
        else
            img=strcat(strcat('0',int2str(i-1)),'.bmp');    
        end
            I{i}=im2bw(imread(img),graythresh(imread(img)));
            Edge{i}=I{i}(:,1:71:72);
    end

    %row1,row2,row3 and row4 are selected pictures for determining each line
    %beginning pixal.
    
    row1=rowDetect('003.bmp');
    row2=rowDetect('010.bmp');
    row3=rowDetect('012.bmp');
    row4=rowDetect('014.bmp');
    
    for i=1:28
        row(i)=min([row1(i) row2(i) row3(i) row4(i)]);
    end
   
end

