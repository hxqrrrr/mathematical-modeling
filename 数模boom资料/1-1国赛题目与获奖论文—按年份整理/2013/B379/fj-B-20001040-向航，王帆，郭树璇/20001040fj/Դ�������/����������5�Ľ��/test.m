A=imread('019.bmp');
B=imread('083.bmp');
C=imread('088.bmp');

A=im2bw(A,graythresh(A));
B=im2bw(B,graythresh(B));
C=im2bw(C,graythresh(C));

count=1;
for i=1:72
    
    for j=180:-1:1
        
        if A(j,i)==0
           distance(count)=j;
           count=count+1;
           break; 
        end
        
    end
    
end
a=mode(distance);
clear distance;

count=1;
for i=1:72
    
    for j=180:-1:1
        
        if B(j,i)==0
           distance(count)=j;
           count=count+1;
           break; 
        end
        
    end
    
end
b=mode(distance);
clear distance;

count=1;
for i=1:72
    
    for j=180:-1:1
        
        if C(j,i)==0
           distance(count)=j;
           count=count+1;
           break; 
        end
        
    end
    
end
c=mode(distance);