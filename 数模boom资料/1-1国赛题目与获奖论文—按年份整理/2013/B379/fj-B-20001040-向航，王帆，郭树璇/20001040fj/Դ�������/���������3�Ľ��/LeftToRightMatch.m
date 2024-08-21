function [ result,leftHeadPool,rightHeadPool,picturePool ] = LeftToRightMatch( Head,Edge,sequence,leftHeadPool,rightHeadPool,picturePool,pictureHemp,rightHemp,picType,picDistance)
%LeftToRightMatch is to matching the one line with the beginning picture.   
tao=25;
start=Edge{1,Head};
n=length(find(sequence~=0));
pairA=Head;
result(1)=Head;
number=2;
M=length(picturePool);

for i=1:n
    N=length(find(sequence~=0));
    M=length(find(picturePool~=0));
    error=[];
    for j=1:N
    
        temp=Edge{1,sequence(j)};
        pairB=sequence(j);
        [degree1,degree2,degree3]=errorDegree(start,temp,tao);
        error=[error;pairA pairB degree1 degree2 degree3];
        
    end
    
    errorUpper=length(error);
    
    for k=1:M
        temp=Edge{1,picturePool(k)};
        pairB=picturePool(k);
        [degree1,degree2,degree3]=errorDegree(start,temp,tao);
        error=[error;pairA pairB degree1 degree2 degree3];
      
    end
    
    errorLength=length(error);
    min1=min(error(:,3));
    min2=min(error(:,4));
    min3=min(error(:,5));
    newError=[];
    for ie=1:errorLength
        
        if error(ie,3)~=0
          temp1=(error(ie,3)-min1)/error(ie,3);
        else
          temp1=0;
        end
        
        if error(ie,4)~=0
            temp2=(error(ie,4)-min2)/error(ie,4);
        else
           temp2=0; 
        end
        
        if error(ie,5)~=0
            temp3=(error(ie,5)-min3)/error(ie,5);
        else
            temp3=0;
        end
        
        newError=[newError;error(ie,1:2) temp1+temp2+temp3];
    
    end
    
    

    
    temp=(find(newError(:,3)==min(newError(:,3))));
    if temp(1,1)<=errorUpper
        result(number)=newError(temp(1,1),2);
        pairA=result(number);
        start=Edge{1,result(number)};
        sequence=dropPool(sequence,newError(temp(1,1),2));
        number=number+1;
        
    else
        
        if picType(result(number-1),1)==picType(newError(temp(1,1),2),1) || picType(result(number-1),2)==picType(newError(temp(1,1),2),2)
             if  abs(picDistance(result(number-1),1)-picDistance(newError(temp(1,1),2),1))<5   || abs(picDistance(result(number-1),2)-picDistance(newError(temp(1,1),2),2))<5
                    result(number)=newError(temp(1,1),2);
                    pairA=result(number);
                    start=Edge{1,result(number)};
                    picturePool=dropPool(picturePool,newError(temp(1,1),2));
                    number=number+1;
                    n=n+1;
             end
        end
    
    end

end
leftHeadPool=dropPool(leftHeadPool,Head);
%% Test the total number of the line and try to match the whole line. 
   
    if length(result)==18
        n=length(find(pictureHemp(12:22,1)==0));
        beforeEnd=Edge{1,result(length(result))};
        pairA=result(length(result));
        error=[];
        for i=1:n
            temp=find(pictureHemp(12:22,1)==0);
            End=Edge{1,rightHemp(temp(i,1))};
            pairB=rightHemp(temp(i,1));
            error=[error;pairA pairB errorDegree(beforeEnd,End,tao)];
        end
        
        temp=(find(error(:,3)==min(error(:,3))));
        result(number)=error(temp(1,1),2);
        rightHeadPool=dropPool(rightHeadPool,result(number));
    end
    
end

