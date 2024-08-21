function [ result,leftHeadPool,rightHeadPool,picturePool ] = RightToLeftMatch( Head,Edge,sequence,leftHeadPool,rightHeadPool,picturePool,pictureHemp,leftHemp,picType,picDistance)
%LeftToRightMatch is to matching the one line with the beginning picture.   
tao=25;
start=Edge{1,Head};
n=length(find(sequence~=0));
pairB=Head;
result(19)=Head;
number=18;
M=length(picturePool);
for i=1:n
    N=length(find(sequence~=0));
    M=length(find(picturePool~=0));
    error=[];
    for j=1:N
    
        temp=Edge{1,sequence(j)};
        pairA=sequence(j);
        [degree1,degree2,degree3]=errorDegree(temp,start,tao);
        error=[error;pairA pairB degree1 degree2 degree3];
        
    end
    
    errorUpper=length(error);
    
    for k=1:M
        temp=Edge{1,picturePool(k)};
        pairA=picturePool(k);
        [degree1,degree2,degree3]=errorDegree(temp,start,tao);
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
        result(number)=newError(temp(1,1),1);
        pairB=result(number);
        start=Edge{1,result(number)};
        sequence=dropPool(sequence,result(number));
        number=number-1;
    else

         if picType(result(number+1),1)==picType(newError(temp(1,1),1),1) || picType(result(number+1),2)==picType(newError(temp(1,1),1),2)
             if  abs(picDistance(result(number+1),1)-picDistance(newError(temp(1,1),1),1))<5   || abs(picDistance(result(number+1),2)-picDistance(newError(temp(1,1),1),2))<5
                    
                    result(number)=newError(temp(1,1),1);
                    pairB=result(number);
                    start=Edge{1,result(number)};
                    picturePool=dropPool(picturePool,newError(temp(1,1),1));
                    number=number-1;
                    n=n+1;
                  
             end
        end
       
    end
    

end
leftHeadPool=dropPool(leftHeadPool,Head);
%% Test the total number of the line and try to match the whole line. 
   
    if length(result)==18
        n=length(find(pictureHemp(1:11,1)==0));
        beforeStart=Edge{1,result(length(result))};
        pairB=result(20-length(result));
        error=[];
        for i=1:n
            temp=find(pictureHemp(1:11,1)==0);
            Start=Edge{1,leftHemp(temp(i,1))};
            pairA=leftHemp(temp(i,1));
            error=[error;pairA pairB errorDegree(Start,beforeStart,tao)];
        end
        
        temp=(find(error(:,3)==min(error(:,3))));
        result(number)=error(temp(1,1),1);
        leftHeadPool=dropPool(leftHeadPool,result(number));
    end
    
end

