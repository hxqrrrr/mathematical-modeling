%�����ǹ�����
%����
l=120;
%ʱ��
s1=[35	39	20	18	55	73	65];
s2=[15	20	15	25	66	54	48];
%�ٶ�
v1=l./s1;
v2=l./s2;
%ƽ���ٶ�
ave_v1=sum(v1)/length(v1);
ave_v2=sum(v2)/length(v2);
%����
s_v1=sum((v1-ave_v1).^2)/length(v1);
s_v2=sum((v2-ave_v2).^2)/length(v2);
%�������й����������
%һ����С�γ�
cs1=[32	34	28	46	83	61	54];
cs2=[13	9	15	20	13	16	21];
%�ٶ�
cv1=l./cs1;
cv2=l./cs2;
%ƽ���ٶ�
cave_v1=sum(cv1)/length(cv1);
cave_v2=sum(cv2)/length(cv2);
%����
cs_v1=sum((cv1-cave_v1).^2)/length(cv1);
cs_v2=sum((cv2-cave_v2).^2)/length(cv2);




