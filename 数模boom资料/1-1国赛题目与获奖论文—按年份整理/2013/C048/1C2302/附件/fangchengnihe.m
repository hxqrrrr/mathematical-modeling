clear
clc
a1=[566.6647541	566.719621	567.0685728	567.0981651	567.129277	567.1576274	567.1912125	567.2121136	567.2277048	567.2485381	567.275333	567.3145249	567.1480366	567.2640516
522.7104513	522.6683532	523.1315794	523.0805765	523.0266955	522.9688285	522.9451814	522.9069293	522.8612449	522.8279637	522.7648961	522.7034767	522.9165376	522.2540994
1.795883731	7.315430375	12.16945526	16.63200668	21.21017887	25.69485936	29.46759953	33.00640665	36.47198733	39.84115365	43.97406515	48.23737841	52.34824346	55.10854517
];
x=a1(1,:);
y=a1(2,:);
z=a1(3,:);
x1=polyfit(z,x,2)%z为自变量，x为因变量
y1=polyfit(z,y,2)%z为自变量，y为因变量
a2=[566.6649138	566.7205038	567.0669496	567.096003	567.1359905	567.1661683	567.1910401	567.2120597	567.2316135	567.2479282	567.2791255	567.3194945	567.1558047	567.2543
522.71021	522.6674349	523.1247923	523.0713536	523.0306902	522.9729661	522.9358895	522.8969708	522.8552903	522.8143306	522.7561448	522.6956723	522.9103383	522.2366
1.768431299	7.31997949	12.13309649	16.5764251	21.24941238	25.73788779	29.42096812	32.95882726	36.45747064	39.78038741	43.9525136	48.22369429	52.33128029	55.11965447
];
x=a2(1,:);
y=a2(2,:);
z=a2(3,:);
x2=polyfit(z,x,2)%z为自变量，x为因变量
y2=polyfit(z,y,2)%z为自变量，y为因变量
a3=[566.7267495	566.7639434	566.3020121	566.3562649	566.4003561	566.4945693	566.5610123	566.621769	566.6913999	566.7680067	566.8228058	566.8940233	566.9594392	567.336
522.7013265	522.6692607	522.9430277	522.9015833	522.8664601	522.8077608	522.7543022	522.7117298	522.6609763	522.5903612	522.5445498	522.4892072	522.4358614	522.2148
1.733590761	7.289610374	12.13556776	16.55391101	21.17783489	25.71187294	29.42734675	32.95110788	36.44851117	39.75847224	43.97136348	48.22231081	52.38757328	55.091
];
x=a3(1,:);
y=a3(2,:);
z=a3(3,:);
x3=polyfit(z,x,2)%z为自变量，x为因变量
y3=polyfit(z,y,2)%z为自变量，y为因变量
a4=[566.7270236	566.7642048	566.304057	566.3507042	566.3987385	566.4978468	566.5623932	566.6202149	566.6938957	566.7597735	566.8260041	566.9049385	566.9658136	567.3375
522.7015402	522.669008	522.9422186	522.9047493	522.8671197	522.8051322	522.7530688	522.7126508	522.658952	522.5957035	522.5420825	522.4820427	522.4312313	522.2135
1.803132147	7.280740547	12.116492	16.57971053	21.18259149	25.68966564	29.41543821	32.96482276	36.4187957	39.79961701	43.95057144	48.16532711	52.36143352	55.087
];
x=a4(1,:);
y=a4(2,:);
z=a4(3,:);
x4=polyfit(z,x,2)%z为自变量，x为因变量
y4=polyfit(z,y,2)%z为自变量，y为因变量