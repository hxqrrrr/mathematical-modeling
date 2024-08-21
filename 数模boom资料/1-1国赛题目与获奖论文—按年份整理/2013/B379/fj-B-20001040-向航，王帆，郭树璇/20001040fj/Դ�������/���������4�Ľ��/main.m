clc,clear
tic
n=209;
[image,I,Edge,Distance,grayEdge]=match(n);
%% Find the picture with left and right blank.
 % 
countLeft=1;
countRight=1;

 for i=1:n
       tempEdge=Edge{i};
    if length(find(tempEdge(:,1))==1)==180 && Distance(i,1)>11
        leftHemp(countLeft)=i;
        countLeft=countLeft+1;
        continue;
    end
    
 end
 

 %% Find the number of pixal of each line left and right edge in picture.
 % count is a cell for storage of each line left and right edge pixal
 % number.
  
 picType=[];
 picDistance=[];
for i=1:n
    
    [A,B]=pictureAttribute(I{i});
    picType=[picType;A];
    picDistance=[picDistance;B];
 
end


 %% Find the each line 
 
% pair=[];
% for i=1:length(leftHemp)

%     jcount=1;
%     flag=0;
%     while(jcount<=length(rightHemp) && flag==0)

%         lineDis1=picDistance(leftHemp(i),1)-picDistance(rightHemp(i),1);
%         lineDis2=picDistance(leftHemp(i),2)-picDistance(rightHemp(i),2);

%         if lineDis1<=3 && lineDis2<=3
%             pair=[pair;leftHemp(i) rightHemp(jcount)];
%             flag=1;
%         end
%         jcount=jcount+1;
 
%     end
          
% end
%% 
%Create the remaining pool
picturePool=[1:209]; 
nleft=length(leftHemp);
 for i=1:nleft
     picturePool=dropPool(picturePool,leftHemp(i));
 end

%
totalPool=length(picturePool);
pictureHemp=zeros(11,19);
pictureCount=zeros(11,1)+1;
count=totalPool*20;
while (count~=0)
    tic
    i=ceil(length(picturePool)*rand());
    error=[];
    for j=1:nleft
       error=[error;leftHemp(j) picturePool(i) (picType(leftHemp(j),1)==picType(picturePool(i),1) && picType(leftHemp(j),2)==picType(picturePool(i),2)) abs(picDistance(leftHemp(j),1)-picDistance(picturePool(i),1))+abs(picDistance(leftHemp(j),2)-picDistance(picturePool(i),2))];
    end
    
    temp=(find(error(:,4)==min(error(:,4))));
    ti=length(temp);
    
    for k=1:ti  
    % for j=1:nleft   
        if  error(temp(k,1),3)==1
            pictureHemp(temp(k,1),pictureCount(temp(k,1),1))=picturePool(i);
            pictureCount(temp(k,1),1)=pictureCount(temp(k,1),1)+1;
            picturePool=dropPool(picturePool,picturePool(i));
            break;
        end
     %end
    end
    toc
    count=count-1;
end

%% First step is to find the shred with blank left.
leftHeadPool=leftHemp;

for i=1:nleft
    
        Head=leftHemp(i);
        sequence=pictureHemp(i,:);
        
        [result,leftHeadPool,picturePool] = LeftToRightMatch( Head,image,grayEdge,sequence,leftHeadPool,picturePool,picType,picDistance);
        leftResult{i}=result;
    
end

%% Second step is to find the proper picture for the shred with blank left.

start=count{1,result(1)};
picturePool=dropPool([1:19],result(1));
pairA=result(1);
number=2;
tempSimilarity=zeros;
while (~isempty(picturePool))
    
    tempSimilarity=[];
    for k=1:length(picturePool)
        pairB=picturePool(k);
        tempPic=count{1,picturePool(k)};
        similarity=similar(start,tempPic);
        tempSimilarity=[tempSimilarity;pairA,pairB,similarity];
    end
    result(number)=tempSimilarity(find(tempSimilarity(:,3)==max(tempSimilarity(:,3))),2);
    start=count{1,result(number)};
    picturePool=dropPool(picturePool,result(number));
    pairA=result(number);
    number=number+1; 
end
toc