clear
L=95.5*2;D=100;d=5;hL=L/2;R=D/2; %ľ�峤������ľ�����볤��Բ����뾶 
ye=-R+d/2:d:R-d/2; xe=sqrt(R^2-ye.^2); 
%�۵����y���꣬x���꣬��20��
%ye=-R+d/2:d:R-d/2; xe=R-abs(ye);
legL=hL-xe;hH=legL(1)/2;ddeg=2;   %�ȳ��ȣ�20��;��Ȱ볤; �Ƕ����� 
 
Tx=[xe -xe;xe -xe];Tx=Tx(:);Tz=zeros(size(Tx));                 %��������
Ty=[ye-d/2 fliplr(ye)+d/2;ye+d/2 fliplr(ye)-d/2];Ty=Ty(:);
legx=[hL*ones(size(xe));hL*ones(size(xe));xe;xe];                %��������     
legy=[ye-d/2;ye+d/2;ye+d/2;ye-d/2];legz=zeros(size(legx)); 
zhoux=[hL-legL(1)/2;hL-legL(1)/2];zhouy=[-R R];zhouz=[0;0];     %�ֽ�������
yb=linspace(ye(1),ye(end),50);xb=sqrt(R^2-yb.^2);
Bx=hL*ones(size(xb)); By=yb; Bz=zeros(size(xb));              %�ȼ���������        
                           
figure(1),clf;hold on
h1=patch(Tx,Ty,Tz,'facecolor',[0.5 0.5 0],'edgecolor',[1 1 1]);%������
h2=patch(legx,legy,legz,'facecolor',[0.2 0.5 0],'edgecolor',[1 1 1]);%������
h3=patch(-legx,legy,legz,'facecolor',[0.1 0.5 0],'edgecolor',[1 1 1]);%������
h4=plot3(zhoux,zhouy,zhouz,'c');h5=plot3(-zhoux,zhouy,zhouz,'c');%���ֽ���
h6=plot3(Bx,By,Bz,'k');h7=plot3(-Bx,By,Bz,'k');%�ȼ�����
hold off;view(3);axis equal;axis([-hL hL -R R 0 2*hH]);axis off;

%for deg=0:ddeg:63.5         %�������������۵��Ƕ�
  deg=63.5/7*7;
    zz=-hH*sind(deg);xz=xe(1)+hH*cosd(deg); %�ֽ���,z�����x����
    alldeg=atan2(-zz*ones(size(xe)),xz-xe); %ÿ�������۵��Ƕ�,20��
    allx=legL.*cos(alldeg)+xe;           %ÿ����ĩ��x���꣬20��
    allz=-legL.*sin(alldeg);            %ÿ����ĩ��z���꣬20��
    alldeg2=atan2(-zz*ones(size(xb)),xz-xb); 
    Bx=(hL-xb).*cos(alldeg2)+xb;Bz=-(hL-xb).*sin(alldeg2);%�ȼ�����x���� 
    minz=min(Bz);                        %�����z���꣬���ӵ�ǰ�߶�
    legx=[allx;allx;xe;xe];                               %��������  
    legz=[allz;allz;zeros(size(allz));zeros(size(allz))]-minz;
    set(h1,'ZData',-minz*ones(size(Tz)));
    set(h2,'XData',legx,'ZData',legz);set(h3,'XData',-legx,'ZData',legz);
    set(h4,'XData',[xz;xz],'ZData',[zz;zz]-minz);
    set(h5,'XData',-[xz;xz],'ZData',[zz;zz]-minz);
    set(h6,'XData',Bx,'ZData',Bz-minz);set(h7,'XData',-Bx,'ZData',Bz-minz);
    pause(0.1);drawnow;
%end
caochang=sqrt((xe-xe(1)).^2+hH.^2-2*hH.*(xe-xe(1)).*cos(alldeg))-(legL-hH);