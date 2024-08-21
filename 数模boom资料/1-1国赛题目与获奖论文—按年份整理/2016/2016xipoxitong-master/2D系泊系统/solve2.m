%% �ļ�˵��
% �˳������������������
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ������1�ļ����£�
% ���㺣�����Ϊ36m/sʱ��Ͱ�͸��ڸֹܵ���б�Ƕȡ�
% ê����״�͸�����ζ�����
% ������������������ʹ�ø�Ͱ����б�ǶȲ�����5�ȣ�
% ê����ê���뺣���ļнǲ�����16�ȡ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%% ����m_qiu��y0 ��x0 ��alpha1��alpha2֮��Ĺ�ϵͼ
clc
clear 
%����
v_wind = 36;
%����
H = 18;
N = 500;
x0 = 20;
I = 2;
L = 22.05;
y0_yn_figure = 0;
xitong_figure = 0;
%����
m_qiu = linspace(1000, 6000, 100);
besty0 = zeros(size(m_qiu));
bestx0 = zeros(size(m_qiu));
alpha1 = zeros(size(m_qiu));
alpha2 = zeros(size(m_qiu));
for i = 1:length(m_qiu)
    [besty0(i), bestx0(i)] = bestpoint(H, N, x0, v_wind, m_qiu(i), I, L, y0_yn_figure);
    y0 = besty0(i);
    x0 = bestx0(i);
    [~, ~, ~, ~, stat] = For2D(y0, x0, v_wind, m_qiu(i), I, L, xitong_figure);
    alpha1(i) = stat.alpha1;
    alpha2(i) = stat.alpha2;
end
%��ͼ
figure(1)
plot(m_qiu, abs(besty0), 'r*-')
xlabel('����������')
ylabel('��ˮ���')
title('��ˮ���h�������������仯����')
title('')
figure(2)
plot(m_qiu, bestx0, 'c<-')
xlabel('����������')
ylabel('��Ư������')
title('��Ư�������������������仯����')

figure(3)
plot(m_qiu, alpha1, 'bo-')
xlabel('����������')
ylabel('��Ͱ��ֱ�н�')
title('��Ͱ��ֱ�н��������������仯����')

figure(4)
plot(m_qiu, alpha2, 'gs-')
xlabel('����������')
ylabel('ê���׶�ˮƽ�н�')
title('ê���׶�ˮƽ�н��������������仯����')

%ע����ò�ֵor���һ�¡�

%% ȷ��m_qiu��ȡֵ��Χ
%��Ͱ����б�ǶȲ�����5�ȣ�ê����ê���뺣���ļнǲ�����16��
alpha1_max = 5;
alpha2_max = 16;

[~, ind1] = min(abs(alpha1 - alpha1_max));
m1 = m_qiu(ind1);
[~, ind2] = min(abs(alpha2 - alpha2_max));
m2 = m_qiu(ind2);

%��Ư��ȫû��ˮ�У�h = 2��
ind3 = min(find(abs(besty0) == 2));%ind3 = find(abs(besty0) == 2, 1)
m3 = m_qiu(ind3);

% s.t.  max{m1, m2}  <  m  < m3

%%  IENSGAii �������m_qiu��ʹ��h��pi*x0^2��alpha1��С����alpha1��2�ڷ�Χ��
fitnessfcn = @multi_GA_m;   
nvars = 1;                     
lb = max([m1, m2]);                  
ub = m3;                     
A = []; b = [];                 
Aeq = []; beq = [];             
options = gaoptimset('ParetoFraction', 0.3, 'PopulationSize', 100, 'Generations', 100, 'StallGenLimit', 100, 'PlotFcns', {@gaplotpareto, @gaplotbestf});

[x_m, fval] = gamultiobj(fitnessfcn, nvars, A, b, Aeq, beq, lb, ub, options);













