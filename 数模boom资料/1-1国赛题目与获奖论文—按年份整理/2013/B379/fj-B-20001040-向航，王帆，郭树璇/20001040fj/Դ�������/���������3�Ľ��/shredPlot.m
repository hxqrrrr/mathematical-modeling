function [  ] = shredPlot( sequence )
%shredPlot is to draw the sequence of shreded picture.

    n=length(find(sequence~=0));
    sequenceTemp=find(sequence~=0);
    Fisrt=sequenceTemp(1,1)-1;
    temp=[];
    for i=1:n
        if sequence(i+Fisrt)<=10
            img=strcat(strcat('00',int2str(sequence(i+Fisrt)-1)),'.bmp');
        elseif sequence(i+Fisrt)<=100
            img=strcat(strcat('0',int2str(sequence(i+Fisrt)-1)),'.bmp');    
        else
            img=strcat(int2str(sequence(i+Fisrt)-1),'.bmp');
        end
        temp=[temp imread(img)];
    end
    imshow(temp);
end

