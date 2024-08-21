function [image,I,Edge,Distance,grayEdge] = match( Length )
%match is to get the edge of picture and get the each line beginning pixal.
%   Length is the number of dataset.
%   Edge is the picture edge.
%   row is the beginning of each line.

    n=Length;
    Distance=[];
    for i=1:n
        if i<=10
            img=strcat(strcat('00',int2str(i-1)),'.bmp');
        elseif i<=100
            img=strcat(strcat('0',int2str(i-1)),'.bmp');    
        else
            img=strcat(int2str(i-1),'.bmp');
        end
        image{i}=imread(img);
        grayEdge{i}=int16(image{i}(:,1:71:72));
        I{i}=im2bw(imread(img),graythresh(imread(img)));
        
        Edge{i}=I{i}(:,1:71:72);
        
        Distance=[Distance;dis(I{i})];
        
    end
    

    
    
    
    %row1,row2,row3 and row4 are selected pictures for determining each line
    %beginning pixal.
    
    %row1=rowDetect('002.bmp');
    %row2=rowDetect('006.bmp');
    %row3=rowDetect('007.bmp');
    
    %for i=1:28
    %    row(i)=min([row1(i) row2(i) row3(i)]);
    %end
   
end

