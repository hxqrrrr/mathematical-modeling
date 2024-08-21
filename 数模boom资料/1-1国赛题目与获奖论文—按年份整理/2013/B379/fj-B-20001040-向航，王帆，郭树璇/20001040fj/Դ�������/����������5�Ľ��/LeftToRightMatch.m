function [ result,leftHeadPool,picturePool ] = LeftToRightMatch( Head,image,Edge,sequence,leftHeadPool,picturePool,picType,picDistance)
%LeftToRightMatch is to matching the one line with the beginning picture.   
tao=1;
start=Edge{1,Head};
n=1000;
pairA=Head;
result(1)=Head;
number=2;
M=length(picturePool);
tempPicPool=picturePool;
tempSequence=sequence;
for i=1:n
    N=length(find(tempSequence~=0));
    M=length(find(tempPicPool~=0));
    error=[];
    for j=1:N
    
        temp=Edge{1,tempSequence(j)};
        pairB=tempSequence(j);
        [degree1,degree2,degree3]=errorDegree(start,temp,tao);
        error=[error;pairA pairB degree1 degree2 degree3];
        
    end
    
    errorUpper=length(error);
    
    for k=1:M
        temp=Edge{1,tempPicPool(k)};
        pairB=tempPicPool(k);
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
  %  if length(temp)==1
    
        if temp(1,1)<=errorUpper
             
            if newError(temp(1,1),3)<=0.3
                
            result(number)=newError(temp(1,1),2);
            pairA=result(number);
            start=Edge{1,result(number)};
            tempSequence=dropPool(tempSequence,newError(temp(1,1),2));
            sequence=dropPool(sequence,newError(temp(1,1),2));
            number=number+1;
            tempSequence=[];
            tempSequence=sequence;
            
            else
                
              subplot(1,2,1),imshow(image{newError(temp(1,1),1)});
              subplot(1,2,2),imshow(image{newError(temp(1,1),2)});
              
              prompt = {'和哪一个在同一行吗？0 means No.'};
              dlg_title = '人工判断';
              num_lines = 1;
              def = {'0'};
              answer = inputdlg(prompt,dlg_title,num_lines,def);
              flag=str2num(answer{1,1});
              if flag~=0
                  result(number)=newError(temp(1,1),2);
                  pairA=result(number);
                  start=Edge{1,result(number)};
                  tempSequence=dropPool(tempSequence,newError(temp(1,1),2));
                  sequence=dropPool(sequence,newError(temp(1,1),2));
                  number=number+1;
                  tempSequence=[];
                  tempSequence=sequence;
                  continue;
              else
                  tempSequence=dropPool(tempSequence,newError(temp(1,1),2));
                  continue;
              end
                  
          end
                

        else
        
            if picType(result(number-1),1)==picType(newError(temp(1,1),2),1) && picType(result(number-1),2)==picType(newError(temp(1,1),2),2)
                 if  abs(picDistance(result(number-1),1)-picDistance(newError(temp(1,1),2),1))<3   && abs(picDistance(result(number-1),2)-picDistance(newError(temp(1,1),2),2))<3
                         subplot(1,2,1),imshow(image{newError(temp(1,1),1)});
              subplot(1,2,2),imshow(image{newError(temp(1,1),2)});
              
              prompt = {'和哪一个在同一行吗？0 means No.'};
              dlg_title = '人工判断';
              num_lines = 1;
              def = {'0'};
              answer = inputdlg(prompt,dlg_title,num_lines,def);
              flag=str2num(answer{1,1});
              if flag~=0
                  result(number)=newError(temp(1,1),2);
                  pairA=result(number);
                  start=Edge{1,result(number)};
                  tempPicPool=dropPool(tempPicPool,newError(temp(1,1),2));
                  picturePool=dropPool(picturePool,newError(temp(1,1),2));
                  number=number+1;
                  tempPicPool=[];
                  tempPicPool=picturePool;
                  continue;
              else
                  tempPicPool=dropPool(tempPicPool,newError(temp(1,1),2));
                  continue;
              end
                     
                                        
                 else
                    if newError(temp(1,1),3)<0.3
                    subplot(1,2,1),imshow(image{newError(temp(1,1),1)});
                    subplot(1,2,2),imshow(image{newError(temp(1,1),2)});
              
                    prompt = {'和哪一个在同一行吗？0 means No.'};
                    dlg_title = '人工判断';
                    num_lines = 1;
                    def = {'0'};
                    answer = inputdlg(prompt,dlg_title,num_lines,def);
                    flag=str2num(answer{1,1});
                  if flag~=0
                      result(number)=newError(temp(1,1),2);
                      pairA=result(number);
                      start=Edge{1,result(number)};
                  
                      tempPicPool=dropPool(tempPicPool,newError(temp(1,1),2));
                      picturePool=dropPool(picturePool,newError(temp(1,1),2));
                      number=number+1;
                                              tempPicPool=[];
                        tempPicPool=picturePool;
                      continue;
                  else
                      tempPicPool=dropPool(tempPicPool,newError(temp(1,1),2));
                      continue;
                  end
                     else
                      tempPicPool=dropPool(tempPicPool,newError(temp(1,1),2));
                      continue;
                  end  
                     
                     
                 end
              
            end
   
            
    %else
            
   %         pictureN=length(temp)+1;
   %         subplot(1,pictureN,1),imshow(image{Head});
   %         for picI=1:(pictureN-1)
    %            subplot(1,pictureN,picI+1),imshow(image{newError(temp(picI,1),2)});
     %       end
      %       prompt = {'和哪一个在同一行吗？0 means No.'};
       %      dlg_title = '人工判断';
        %     num_lines = 1;
        %     def = {'0'};
        %     answer = inputdlg(prompt,dlg_title,num_lines,def);
        %     flag=str2num(answer{1,1});
        %     if flag~=0
        %         result(number)=newError(temp(1,flag),2);
        %         pairA=result(number);
        %         start=Edge{1,result(number)};
        %         picturePool=dropPool(picturePool,result(number));
        %         tempPicPool=dropPool(tempPicPool,result(number));
        %         number=number+1;
        %         continue;
        %     else
        %         for deletei=1:length(temp)
        %         tempPicPool=dropPool(tempPicPool,newError(temp(deletei,1),2));
        %         end
            % end
            
            %  if newError(temp(1,1),3)<0.2
            %            result(number)=newError(temp(1,1),2);
            %            pairA=result(number);
            %            start=Edge{1,result(number)};
            %            picturePool=dropPool(picturePool,newError(temp(1,1),2));
            %            tempPicPool=dropPool(tempPicPool,newError(temp(1,1),2));
            %            number=number+1;
            %            n=n+1;
            %         else
            %            tempPicPool=dropPool(tempPicPool,newError(temp(1,1),2));
            %  end
        end
         tempSequence=dropPool(tempSequence,newError(temp(1,1),2));
        
end
    leftHeadPool=dropPool(leftHeadPool,Head); 
    for deletei=1:length(find(sequence>0))
          picturePool=addPool(picturePool,sequence(deletei));
    end
end

