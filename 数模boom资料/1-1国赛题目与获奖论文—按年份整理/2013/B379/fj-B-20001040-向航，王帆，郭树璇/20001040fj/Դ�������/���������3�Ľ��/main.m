clc,clear
tic
n=209;
[image,I,Edge,Distance,grayEdge]=match(n);
%load('answer1.mat');
%tempSequence=[];
%for i=1:length(leftResult)
%    tempSequence=[tempSequence;leftResult{1,i}];
%end
%final=rowSort(tempSequence,image);
%clear tempSequence;
%% Find the picture with left and right blank.
 % 
countLeft=1;
countRight=1;
 for i=1:n
       tempEdge=Edge{i};
    if length(find(tempEdge(:,1))==1)==180 && Distance(i,1)>5
        leftHemp(countLeft)=i;
        countLeft=countLeft+1;
        continue;
    end
    
    if length(find(tempEdge(:,2))==1)==180 && Distance(i,2)>5
        rightHemp(countRight)=i;
        countRight=countRight+1;
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
nright=length(rightHemp);
totalHemp=nleft+nright;
 for i=1:totalHemp
     if i<=length(leftHemp)
     picturePool=dropPool(picturePool,leftHemp(i));
     else
     picturePool=dropPool(picturePool,rightHemp(i-nleft));
     end
 end

%
totalPool=length(picturePool);
pictureHemp=zeros(22,19);
pictureCount=zeros(22,1)+1;
count=totalPool*30;
while (count~=0)
    tic
    i=ceil(length(picturePool)*rand());
    for j=1:totalHemp
        if j<=length(leftHemp)
         if picType(leftHemp(j),1)==picType(picturePool(i),1) && picType(leftHemp(j),2)==picType(picturePool(i),2)
             if  abs(picDistance(leftHemp(j),1)-picDistance(picturePool(i),1))<3   && abs(picDistance(leftHemp(j),2)-picDistance(picturePool(i),2))<3
                 pictureHemp(j,pictureCount(j,1))=picturePool(i);
                 pictureCount(j,1)=pictureCount(j,1)+1;
                 picturePool=dropPool(picturePool,picturePool(i));
                 break; 
             elseif abs(picDistance(leftHemp(j),1)-picDistance(picturePool(i),1))<8   && abs(picDistance(leftHemp(j),2)-picDistance(picturePool(i),2))<8
             subplot(1,2,1),imshow(image{leftHemp(j)});
             subplot(1,2,2),imshow(image{picturePool(i)});
             prompt = {'两者在同一行吗？0 is 不在，1 is 在。'};
             dlg_title = '人工判断';
             num_lines = 1;
             def = {'0'};
             answer = inputdlg(prompt,dlg_title,num_lines,def);
             flag=str2num(answer{1,1});
             if flag==1
                 pictureHemp(j,pictureCount(j,1))=picturePool(i);
                 pictureCount(j,1)=pictureCount(j,1)+1;
                 picturePool=dropPool(picturePool,picturePool(i));
                 break;
             end
             end
            
         end
         
        else
             
         if picType(rightHemp(j-nleft),1)==picType(picturePool(i),1) && picType(rightHemp(j-nleft),2)==picType(picturePool(i),2)
             if  abs(picDistance(rightHemp(j-nleft),1)-picDistance(picturePool(i),1))<3   && abs(picDistance(rightHemp(j-nleft),2)-picDistance(picturePool(i),2))<3
                 pictureHemp(j,pictureCount(j,1))=picturePool(i);
                 pictureCount(j,1)=pictureCount(j,1)+1;
                 picturePool=dropPool(picturePool,picturePool(i));
                 break;
             elseif abs(picDistance(rightHemp(j-nleft),1)-picDistance(picturePool(i),1))<3   && abs(picDistance(rightHemp(j-nleft),2)-picDistance(picturePool(i),2))<3
             
             subplot(1,2,1),imshow(image{picturePool(i)});
             subplot(1,2,2),imshow(image{rightHemp(j-nleft)});
             prompt = {'两者在同一行吗？0 is 不在，1 is 在。'};
             dlg_title = '人工判断';
             num_lines = 1;
             def = {'0'};
             answer = inputdlg(prompt,dlg_title,num_lines,def);
             flag=str2num(answer{1,1});
             if flag==1
                 pictureHemp(j,pictureCount(j,1))=picturePool(i);
                 pictureCount(j,1)=pictureCount(j,1)+1;
                 picturePool=dropPool(picturePool,picturePool(i));
                 break;
             end 
             end
         end
         
        end
        
    end
    toc
    count=count-1;
    
end

%% First step is to find the shred with blank left.
leftHeadPool=leftHemp;
RightHeadPool=leftHemp;

for i=1:totalHemp
    if i<=nleft
        Head=leftHemp(i);
        sequence=pictureHemp(i,:);
        
        [result,leftHeadPool,RightHeadPool,picturePool] = LeftToRightMatch( Head,grayEdge,sequence,leftHeadPool,RightHeadPool,picturePool,pictureHemp,rightHemp,picType,picDistance);
        leftResult{i}=result;
    else
        Head=rightHemp(i-nleft);
        sequence=pictureHemp(i,:);
        
        [result,leftHeadPool,RightHeadPool,picturePool] = RightToLeftMatch( Head,grayEdge,sequence,leftHeadPool,RightHeadPool,picturePool,pictureHemp,leftHemp,picType,picDistance);
        rightResult{i-nleft}=result;    
    end
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