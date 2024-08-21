%调用f2函数，输入命令ezplot('f2(t)',[0,600]);输出时间-队长曲线
%第四问差分模型，求解当发生事故时上游路口恰好红灯的情况下，车辆数目与时间的关系
%做出图像，通过车辆数目实际值的分析，得到车辆排队长度将到达上游路口的时间
function m=f2(t) 
y=0;
for k=0:90
    %时间上限为90
    if t<(30+60*k)&&t>=(60*k)
        y=k*5.86-7.46*(t-60*k)/30;
    else
        if t>=(30+60*k)&&t<(60*k+60)
        y=k*5.86-7.46+13.32*(t-30-60*k)/30;
        %13.32-7.46=5.82
        %其中22.86为绿灯时候每半分钟的到达的车辆，2.11为红灯时候每半分钟的到达的车辆
        %半分钟的车辆通行能力为8.9辆，从小区离开的车辆每半分钟为0.67
        %2.11-8.9-0.67=-7.46，22.89-8.9-0.67=13.32       
        else y=y;
        end
    end
m=y;
end

