function [k]=shibie(W,bm,ju,n)

k=find(W(n,:)==min(W(n,:)));
lo=length(k);
if(lo>1)
    k
    for i=1:lo
        s(:,[1:72])=bm{ju(n),1};
        s(:,[73:144])=bm{ju(k(i)),1};
        imwrite(s,'lena.jpg','quality',100); 
        figure;
        imshow('lena.jpg')
    end
    b=input('�˹���Ԥ');
    if(b==0)
        for i=1:lo
            W(n,k)=inf;
        end
        close all;
        [k]=shibie(W,bm,ju,n);
    else
            k=k(b);
    end
end
        s(:,[1:72])=bm{ju(n),1};
        s(:,[73:144])=bm{ju(k),1};
        imwrite(s,'lena.jpg','quality',100); 
        figure;
        imshow('lena.jpg')
        b=input('ʶ��');
        if(b==0)
            W(n,k)=inf;
            close all;
            [k]=shibie(W,bm,ju,n);
        else
            k=k;
            W(n,k)=inf;
        end