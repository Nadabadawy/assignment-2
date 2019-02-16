clc
clear all
close all
% 1st hypothesis using all the features and logistic regression function with error E
ds = datastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',250);
T = read(ds);
 %%%%% cross validation 60%
m_60 = length(T{:,1})*0.6;
m_20 = length(T{:,1})*0.2;
m_40 = length(T{:,1})*0.4;
x=T{1:m_60,1:13};
y=T{1:m_60,14};
 
pos=find(y==1);
neg=find(y==0);
 
%compute cost and gradient decent
m= length(T{1:m_60,1});
a = length(T{:,1});
Alpha=0.01;
lamda=100;
 
X=[ones(m_60,1) x];
Y=T{1:m_60,14}/mean(T{1:m_60,14});


n=length(X(1,:)); 
 for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
 end
 
 theta=zeros(n,1);
 
 h=1./(1+exp(-X*theta)); %hypothesis function according to lecture rule
 
 k=1;
 E(k)=-(1/m_60)*sum(Y.*log(h)+(1-Y).*log(1-h))+(lamda/(2*m_60))*sum((theta).^2);  %cost function
 
 grad=zeros(size(theta,1),1);     %gradient vector
  
 for i=1:size(grad)
     grad(i)=(1/m_60)*sum((h-Y)'*X(:,i));
 end
 

R=1;
while R==1
Alpha=Alpha*1;
theta=theta-(Alpha/m)*X'*(h-Y);
h=1./(1+exp(-X*theta));  %hypothesis function
k=k+1

E(k)=(-1/m)*sum(Y.*log(h)+(1-Y).*log(1-h))+(lamda/(2*m))*sum((theta).^2);
if E(k-1)-E(k) <0 
    break
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.00001
    R=0;
end
end
 
figure(1)
plot(E)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cross validation 20%
xx=T{m_60+1:m_60+m_20,1:13};
yy=T{m_60+1:m_60+m_20,14};
 
 
%compute cost and gradient decent
mm= length(T{m_60+1:m_60+m_20,1});
Alpha=0.01;
lamda=100;
 
XX=[ones(m_20,1) xx];
YY=T{m_60+1:m_60+m_20,14}/mean(T{m_60+1:m_60+m_20,14});


nn=length(XX(1,:)); 
 for w=2:nn
    if max(abs(XX(:,w)))~=0
    XX(:,w)=(XX(:,w)-mean((XX(:,w))))./std(XX(:,w));
    end
 end
 
 theta1=theta;
 
 hh=1./(1+exp(-XX*theta1)); %hypothesis function according to lecture rule
 
 kk=1;
 EE(kk)=-(1/m_20)*sum(YY.*log(hh)+(1-YY).*log(1-hh))+(lamda/(2*m_20))*sum((theta1).^2);  %cost function
 
 grad=zeros(size(theta1,1),1);     %gradient vector
  
 for i=1:size(grad)
     grad(i)=(1/m_20)*sum((hh-YY)'*XX(:,i));
 end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% last cross validation 20%
p=a-(m_60+m_20);
xxx=T{m_60+m_20+1:end,1:13};
yyy=T{m_60+m_20+1:end,14};
 
 
%compute cost and gradient decent
mmm= length(T{m_60+m_20+1:end,1});
Alpha=0.01;
lamda=100;
 
XXX=[ones(p,1) xxx];
YYY=T{m_60+m_20+1:end,14}/mean(T{m_60+m_20+1:end,14});


nnn=length(XXX(1,:)); 
 for w=2:nnn
    if max(abs(XXX(:,w)))~=0
    XXX(:,w)=(XXX(:,w)-mean((XXX(:,w))))./std(XXX(:,w));
    end
 end
 
 theta2=theta1(1:nnn);
 
 hhh=1./(1+exp(-XXX*theta2)); %hypothesis function according to lecture rule
 
 kkk=1;
 EEE(kkk)=-(1/m_20)*sum(yyy.*log(hhh)+(1-yyy).*log(1-hhh))+(lamda/(2*m_20))*sum((theta2).^2);  %cost function
 
 grad=zeros(size(theta2,1),1);     %gradient vector
  
 for i=1:size(grad)
     grad(i)=(1/m_20)*sum((hhh-yyy)'*XXX(:,i));
 end
 

