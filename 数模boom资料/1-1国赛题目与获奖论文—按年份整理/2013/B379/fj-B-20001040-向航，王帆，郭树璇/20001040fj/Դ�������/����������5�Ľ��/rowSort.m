function [ result ] = rowSort( sequence,data )
%% rowSort is to sort the row.    

    number=2;
    %% Get the edge of row.
    [n,~]=size(sequence);
    for i=1:n
            [~,m]=size(sequence);
            tempRow=[];
        for j=1:m
            tempRow=[tempRow data{1,sequence(j)}];            
        end
            Row{i}=tempRow;
    end
    
    %% 

    picType=[];
    picDistance=[];
    for i=1:n
    
        [A,B]=pictureAttribute(intRow{i});
        picType=[picType;A];
        picDistance=[picDistance;B];
 
    end
    
    for i=1:n
        whiteUp=max(picDistance(:,1));
        if picDistance(i,1)==whiteUp
           startRow=i;
           break;
        end
    end
    
    for i=1:n
            [~,m]=size(sequence);
            tempEdge1=[];
            tempEdge2=[];
        for j=1:m
            tempEdge1=[tempEdge1 data{1,sequence(j)}(1,:)];
            tempEdge2=[tempEdge2 data{1,sequence(j)}(180,:)];
        end
            Edge{i}=[tempEdge1;tempEdge2];
    end
    
    
    %%  
    tao=25;
    newSequence=[1:n];
    newSequence=dropPool(newSequence,startRow);
    start=Edge{1,startRow};
    pairA=startRow;
    result(1,1)=startRow;
    for i=1:n
        
        error=[];
        for j=1:length(newSequence)
            temp=Edge{1,j};
            pairB=newSequence(j);
            error=[error;pairA pairB errorDegreeRow(start,temp,tao)]; 
        end
        
        temp=(find(error(:,3)==min(error(:,3))));
        result(number)=error(temp(1,1),2);
        pairA=result(number);
        start=Edge{1,result(number)};
        newSequence=dropPool(newSequence,result(number));
        number=number+1;
    end
   
end