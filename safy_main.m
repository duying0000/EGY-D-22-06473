%% 气象数据读取
clc
clear
clear global
global TTA SSR S RSLAI DayT LAIT
load('meteorological.mat')
%% 输入实测LAI和对应时期
DayT = [202	205	207	210	214	216	227	229	231	233	235];% 播种后天数	
LAIT = [1.388	1.421	1.411	1.753	1.815	1.599	1.525	1.405	1.408	1.332	1.144];% 对应的实测LAI值	

%% 模型参数拟合
x0=[0.8 0.15 0.6];%参数初始值    
bl=[0.001 0.011 0];%参数下限  
bu=[2.5 1 1];%参数上限 
% x0=[0.43 1.96e-3 1008 5875 2];%参数初始值    
% bl=[0.1 0.0001 500 1000 1.5];%参数下限  
% bu=[0.7 0.01 1600 20000 2.5];%参数上限  

X=zeros(1000,3); 
F=zeros(1000,1); 
%     LAIT=LA(:,i)';
%     DayT=DD(:,i)';
    for ii=1:1000
        [X1,F1]=SCE('func',x0,bl,bu);
        X(ii,:)=X1;
        F(ii)=F1;
    end
%   d0=10;pla=median(X(:,1));plb=median(X(:,2));stt=median(X(:,3));rs=median(X(:,4));elue=median(X(:,5));FF=median(F);
    a=median(X(:,1));b=median(X(:,2));g=median(X(:,3));% kk_t=median(X(:,3));f=median(X(:,4));

%% 计算模拟的LAI和Ks
[DAM,LAI,Ks] =safytest_gas2(d0,pla,plb,stt,rs,elue,g,a,b,8,3,TTA,SSR,S);
%%
n = length(DayT);
LL(:,1)=LAI(DayT);
LL(:,2)=LAIT;
y = LL(:,1);
y2 = LL(:,2);
sum=0;
sum1=0;
sum2=0;
sum3=0;
sum4 = 0; % yh
for i=1:n
    sum=sum+(y(i)-y2(i))*(y(i)-y2(i)); % Sum of squares of errors,validation (SSE)
    sum1=sum1+y(i)*y(i);
    sum2=sum2+(y(i)-mean(y))*(y(i)-mean(y));
    sum3=sum3+(y(i)-mean(y2))*(y(i)-mean(y2));
    sum4 = sum4 + abs(y(i)-y2(i))/y(i); % yh
end
RE=1-(sum/sum2); % Defined by Lorenz,1956
CE=1-(sum/sum3); % CE Defined by Briffa.
MSE=sum/n; % Mean squared erro of validation
RMSE=sqrt(sum/n) % Root mean squared error of validation
RelativeError = sum4/n  % yh
[R,P]=corrcoef(y2,y);
Z=R(:,2);
Rsquare=Z(1)*Z(1)
AdjRsquare=1-(1-Rsquare)*(n-1)/(n-1-1)
%% 结果模拟图
figure (1)
plot(DAM),xlim([150 260]),xlabel('days'),ylabel('Aboveground dry biomass (kg·m-2)');

figure (2)
plot(LAI),xlim([150 260]),xlabel('days'),ylabel('LAI')
hold on
plot(DayT,LAIT, 'o')
% hold on
% plot(LAI0),xlim([150 260])
% hold on
% plot(DayT,LAIT, 'o',DayT,LAIT2,'*',178,LAI0(178),'o')
%%
daydis = [150 151 152];
disLAI = LAI(daydis)
