%δ�˹���Ԥǰ��õĸ�ԭͼ���������˹���Ԥ����
clear all;clc;
ht=load('chubu.txt');
load 'tu.mat';
bm=tu;
for i=1:11
    for j=1:19
        AI([(1+(i-1)*180):(180*i)],[(1+(j-1)*72):(72*j)])=bm{ht(i,j),1};
    end
end
for i=1:11
    bmp{i,1}=AI(1+(i-1)*180:180*i,1:72*19);
end

d=[1 2 3 4 5 6 7 8 9 10 11];  %��Ϊͨ���˹���Ԥ�����е�����õ�����ȷ������
for i=1:11
    AI([(1+(i-1)*180):(180*i)],:)=bmp{d(i),1};
end

figure(1)
imwrite(AI,'hangtu.jpg','quality',100); %���ͼ��
imshow('hangtu.jpg')

%�ó�����Ȧ����Ҫ�˹���Ԥ��������ƥ���λ��
% Create rectangle
annotation(figure(1),'rectangle',...
    [0.364836017569546 0.844854070660522 0.152733528550512 0.0384024577572965],...
    'FaceColor','flat');

% Create rectangle
annotation(figure(1),'rectangle',...
    [0.523693997071742 0.660522273425499 0.113202049780381 0.0322580645161289],...
    'FaceColor','flat');

% Create rectangle
annotation(figure(1),'rectangle',...
    [0.364103953147877 0.466973886328725 0.272060029282577 0.0368663594470047],...
    'FaceColor','flat');

% Create rectangle
annotation(figure(1),'rectangle',...
    [0.363371888726208 0.359447004608295 0.0517086383601757 0.0291858678955454],...
    'FaceColor','flat');

% Create rectangle
annotation(figure(1),'rectangle',...
    [0.494411420204978 0.248847926267281 0.0626896046852123 0.032258064516129],...
    'FaceColor','flat');

% Create rectangle
annotation(figure(1),'rectangle',...
    [0.580062957540264 0.112135176651306 0.0553689604685211 0.0291858678955453],...
    'FaceColor','flat');

title('�ó�����Ȧ����Ҫ�˹���Ԥ��������ƥ���λ��');

fprintf('������Ϊ�ٷ�֮54.55\n')

fprintf('�����˹���Ԥ����Ϊ')

ci=ceil(1980*1368*0.5455/(180*72*5))