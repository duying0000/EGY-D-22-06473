function [DAM1,LAI1,Ks1] = safytest_gas2(d0,pla,plb,stt,rs,elue,g,a,b,kk_t,f,TTA,SSR,S)
%     global testday
    dam0 = 5.3; %初始生物量
    lai0 = 0.1; %初始LAI
    c1 = 0.48; %光合有效辐射分数
    k = 0.53; %光拦截系数
    t_min = 0; %小麦生长最低温
    t_max = 26;%小麦生长最高温
    t_opt = 19;%小麦生长最适温，17-23℃
    sla = 0.019;%比叶面积

%     pla = 0.43;%分配参数
%     plb = 0.0007;%分配参数
%     stt = 985.82;%积温阈值
%     rs = 16679.42; %衰老速率
%     d0 = 10; %出苗日期
%     elue = 2; %有效光能利用效率
%     dg_x = 0.5; %与泄漏点水平距离  
%     a = 1; 
%     f = 3; %f为凸形胁迫方程形状因子(参考值1~6)，f值越大，作物对胁迫越不敏感，胁迫方程曲率越大，f值为1时，胁迫方程为线性方程;


    kz = 0.0009; %根系生长速率 m·℃·d-1
    zr0 = 0.1; %初始根系深度
    zrmax = 1.6; %最大根系深度，冬小麦1.5-1.8m
    de_g = 0.6; %天然气泄漏点深度
    day_g = 178; %开始通气日期
    d_g0 = 1.75; %空白对照点距离(m)
    d= length(TTA);
    l=d-d0+1;
    Zr = zeros(l,1);
    Ks = zeros(l,1);
    DAM = zeros(l,1);
    LAI = zeros(l,1);
    DAM1=zeros(d,1);
    LAI1=zeros(d,1);
    Ks1=zeros(d,1);
    DAM(1) = dam0;
    LAI(1) = lai0;
    d0=round(d0);
    Zr(1) = zr0;
    Ks(1)=1;
    for i=d0+1:d
        t = i-d0+1;
        ta = TTA(i);
        rg = SSR(i);
    % DAM
        fap = 1-exp(-k*LAI(t-1));
        if ta > t_min && ta < t_opt
            ft = 1-((t_opt-ta)/(t_opt-t_min))^2;
        else if ta>t_opt && ta<t_max
            ft = 1-((ta-t_opt)/(t_max-t_opt))^2;    
            else
              ft = 0;
            end
        end
        if ta>0
            Zr(t) = Zr(t-1) + kz*ta; % 每天的根系深度
        else
            Zr(t) = Zr(t-1);
        end
        if Zr(t)>=zrmax
            Zr(t) = zrmax;
        end
        dg_y = abs(Zr(t)-de_g);

        dg_x = g*d_g0;
        DG = sqrt(dg_x^2 + (a*dg_y)^2);
        k_g = (d_g0 - DG)/d_g0;
        if k_g < 0
            k_g = 0;
        else if k_g > 1
                k_g = 1;
            end
        end
        if i<day_g
            Ks(t) = 1;
        else
            Ks(t) = 1-(exp(k_g*f)-1)/(exp(f)-1);
        end
        E = elue*Ks(t);
        dam = rg*c1*fap*E*ft;
        DAM(t) = DAM(t-1) + dam;
        
        KK = t-sum(Ks);
        if KK > kk_t
            lai_c = LAI(t-1)*(1-Ks(t))*b;
        else
            lai_c = 0;
        end
        
    %LAI
        smt = S(i)-S(d0-1);
        pl = 1-pla*exp(plb*smt);
        lai_a = dam*pl*sla;
        if smt>stt
            lai_b=LAI(t-1)*(smt-stt)/rs;
        else
            lai_b = 0;
        end
        LAI(t)=LAI(t-1)+lai_a-lai_b-lai_c;
    end  
   DAM1(d0:d)=DAM;
   LAI1(d0:d)=LAI;
   Ks1(d0:d)=Ks;

end