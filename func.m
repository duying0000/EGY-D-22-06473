function f=func(x)
global TTA SSR S RSLAI DayT LAIT LAI1
d0=10;pla=0.6733;plb=1.33e-04;stt=1580;rs=1210;elue=1.698;
a=x(1);b=x(2);kk_t=8;f=3;g=x(3);
RSLAI = LAIT;    
[~,LAI,~] = safytest_gas2(d0,pla,plb,stt,rs,elue,g,a,b,kk_t,f,TTA,SSR,S);
LAI1 = LAI(DayT)';
f=sqrt(sum((LAI1-RSLAI).^2)/length(DayT));
return;