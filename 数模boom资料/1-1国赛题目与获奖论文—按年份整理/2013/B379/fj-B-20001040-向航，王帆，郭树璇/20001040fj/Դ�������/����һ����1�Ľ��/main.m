tic
n=19;
[Edge,row]=match(n);
rowNumber=length(row);
%% Find the number of pixal of each line left and right edge in picture.
 % count is a cell for storage of each line left and right edge pixal
 % number.
for i=1:n
    temp=zeros;
    tempEdge=Edge{i};
    for j=1:rowNumber
        temp(j,1)=length(find(tempEdge(row(j):row(j)+40,1)==0));
        temp(j,2)=length(find(tempEdge(row(j):row(j)+40,2)==0));
    end
    count{i}=temp;
end

%% First step is to find the shred with blank left.
 result=zeros(19);
for j=1:n
    if sum(count{1,j}(:,1))==0
        result(1)=j;
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