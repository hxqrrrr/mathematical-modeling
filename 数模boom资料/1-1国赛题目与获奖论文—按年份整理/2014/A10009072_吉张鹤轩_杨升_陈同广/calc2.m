function differ = calc(fit)

%fit = [1.47201781595905e-15,-1.45018635953839e-12,5.24789656694673e-10,-7.62445538325332e-08,4.61307014593174e-06,-0.000104467238973322,0.112262400663209];
%[1.4e-15 -1.5e-12 5.2e-10 -8e-8 4.55e-6 -1e-3 0]
%[1.5e-15 -1.4e-12 5.3e-10 -7e-8 4.65e-6 1e-3 0.2]

%[1.46566290179000e-15,-1.46157635173758e-12,5.27033346085919e-10,-7.71007791330307e-08,4.56471413910160e-06,7.39479694412659e-05,0.0979452381207574]
%[1.31909661161100e-15,-1.60773398691134e-12,4.74330011477327e-10,-8.48108570463338e-08,4.10824272519144e-06,6.65531724971393e-05,0.0881507143086817]
%[1.61222919196900e-15,-1.31541871656382e-12,5.79736680694511e-10,-6.93907012197276e-08,5.02118555301176e-06,8.13427663853925e-05,0.107739761932833]


T = 0.1;
t = 0;
x = 0; y = 0;
x2 = 0; y2 = 0;

Q = 6.4;

sita = fit(7);
vx = 1692;
vy = 0;

m0 = 2400;
m = 2400;

N_altered = 7500;
N = N_altered / m0;
G = 3844.6 / m0;
r = 1749372;
H = 15000;
ax = N - vy*vx/r;
ay = G - vx^2/r;

%hold on
% history = [];
i = 0;
while (y>-12000 && m>1000)
    x2 = x + vx*T - 0.5*ax*T;
    y2 = y - vy*T - 0.5*ay*T;
    vx = vx - ax*T; 
%     if vx < 0
%         vx = 0;
%     end;
    vy = vy + ay*T;
    
    r2 = 1749372 + y2;
    G = 3844.6/m0/(r^2)*(r2^2);
    m = m - 2.55/7500*N_altered*T;
    t = t + T;
    N = N_altered / m;
    
    sita = fit(1)*t.^6+fit(2)*t.^5+fit(3)*t.^4+fit(4)*t.^3+fit(5)*t.^2+fit(6)*t+fit(7);
    
    ax = N * cos(sita) - vy*vx/r2;
    ay = G - N * sin(sita) - vx^2/r2;
    
    %plot([x,x2],[y,y2],'r');
    x = x2; y = y2; r = r2;
    
    %history = [history; (i-1)*T y vx vy sita];
    i = i + 1;
end

if (m<=500)
    differ = 200000;
else
    differ = m0 - m;
end

%if (abs(vx) > 10)
%    differ = 200000;
%end

if (abs(sqrt(vx^2+vy^2) - 57) > 1)
    differ = 300000;
end

%[1.35e-15 -1.55e-12 5.15e-10 -8.25e-8 4.5e-6 -1.5e-3 0]
%[1.55e-15 -1.35e-12 5.35e-10 -6.75e-8 4.7e-6 1.5e-3 0.25]







%当水平速度尽可能小的时候，解为下解
%[1.34814486000655e-15,-1.41549730106523e-12,5.34382660183033e-10,-8.18808215848505e-08,4.97635776543674e-06,6.74845654142907e-05,0.103736910112559]
%改为不考虑水平速度一定要接近0
%[1.50459666602573e-15,-1.63733943325256e-12,6.08314652970012e-10,-8.66961374868528e-08,5.10094161874476e-06,7.72276926294689e-05,0.0860318864274153]
%[1.52934053882916e-15,-1.43272091614366e-12,4.99004450285274e-10,-7.13539518999251e-08,4.14432635131618e-06,7.14803605931066e-05,0.0967897293570605]