%��Ϊ���ܺ���������ʶ������ͼ���Ƿ������������ڣ��ɹ�ͬĿ¼�µ��õ����ܵ�������
function [k]=panbie1(W,bm,n)    

k=find(W(n,:)==min(W(n,:)));
lo=length(k);
if(lo>1)  %�����ֶ�����ŵĿ��н�ʱ
    k
    for i=1:lo
        s(:,[1:72])=bm{n,1};
        s(:,[73:144])=bm{k(i),1};
        imwrite(s,'lena.jpg','quality',100); 
        figure;
        imshow('lena.jpg')
    end
    b=input('�˹���Ԥ');  %�ж��Ƿ��������������ڽ�ͼ
    if(b==0)
        for i=1:lo
            W(n,k)=inf;   %����Щ�����������޳�
        end
        close all;
        [k]=panbie(W,bm,n); %ѭ�����ú���
    else
            k=k(b);
    end
end
        s(:,[1:72])=bm{n,1};
        s(:,[73:144])=bm{k,1};
        imwrite(s,'lena.jpg','quality',100); 
        figure;
        imshow('lena.jpg')
        b=input('ʶ��');
        if(b==0)
            W(n,k)=inf;
            close all;
            [k]=panbie(W,bm,n);
        else
            k=k;
            W(n,k)=inf;
        end
