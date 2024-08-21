%以下是公交车
%距离
l=120;
%时间
s1=[35	39	20	18	55	73	65];
s2=[15	20	15	25	66	54	48];
%速度
v1=l./s1;
v2=l./s2;
%平均速度
ave_v1=sum(v1)/length(v1);
ave_v2=sum(v2)/length(v2);
%方差
s_v1=sum((v1-ave_v1).^2)/length(v1);
s_v2=sum((v2-ave_v2).^2)/length(v2);
%公交车有关量计算结束
%一下是小轿车
cs1=[32	34	28	46	83	61	54];
cs2=[13	9	15	20	13	16	21];
%速度
cv1=l./cs1;
cv2=l./cs2;
%平均速度
cave_v1=sum(cv1)/length(cv1);
cave_v2=sum(cv2)/length(cv2);
%方差
cs_v1=sum((cv1-cave_v1).^2)/length(cv1);
cs_v2=sum((cv2-cave_v2).^2)/length(cv2);




