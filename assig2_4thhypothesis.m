clc
clear all
close all
% 3rd hypothesis using the first 4 features and combination of polynomials and logistic regression function with error E
ds = datastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',250);
T = read(ds);
 
x=T{:,1:4};
x1 = T{:,5:10};
y=T{:,14};
 
pos=find(y==1);
neg=find(y==0);
 
%compute cost and gradient decent
m= length(T{:,1});
Alpha=0.01;
lamda=100;
 
X=[ones(m,1) x exp(-x1) exp(-x1.^2) exp(-x1.^3)];
Y=T{1:m,14}/mean(T{1:m,14});


n=length(X(1,:)); 
 for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
 end
 
 theta=zeros(n,1);
 
 h=1./(1+exp(-X*theta)); %hypothesis function according to lecture rule
 
 k=1;
 E(k)=-(1/m)*sum(Y.*log(h)+(1-Y).*log(1-h))+(lamda/(2*m))*sum((theta).^2);  %cost function
 
 grad=zeros(size(theta,1),1);     %gradient vector
  
 for i=1:size(grad)
     grad(i)=(1/m)*sum((h-Y)'*X(:,i));
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
 
figure(2)
plot(E)