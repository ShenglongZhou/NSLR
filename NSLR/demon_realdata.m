clc; close all; clear all

test  = 3;
prob  = {'arcene','colon','duke'};
data  = load(strcat(prob{test},'.mat')); 
class = load(strcat(prob{test},'class.mat'));  

y     = class.y;
y(y  ~= 1)=0;
[n,p] = size(data.X); 
ntp   = 1+(test>2);
X     = Normalization(data.X,ntp); % Normalization is suggested for real data

if test>2
tdata  = load(strcat(prob{test},'_test.mat')); 
tclass = load(strcat(prob{test},'class_test.mat'));
ty     = tclass.y;
ty(ty ~= 1)=0;
tX     = Normalization(tdata.X,ntp);
end
 
out   = NSLR(X,y,ceil(0.01*p));
fprintf(' Training Sparsity:        %d\n', out.sparsity);
fprintf(' Training CPU time:        %.3fsec\n', out.time);
fprintf(' Training Logistic Loss:   %5.2e\n', out.logitloss);
fprintf(' Training Classify Error:  %5.3f\n', out.ser);
fprintf(' Training Sample size:     %dx%d\n', n,p);

if test>2
   T   = find(out.sol);
   Xz  = tX(:,T)*out.sol(T);
   ft  = LogitLoss(Xz,ty);
   ser = nnz(ty-max(0,sign(Xz)))/length(ty); 
   fprintf(' Testing  Logistic Loss:   %5.2e\n', ft);
   fprintf(' Testing  Classify Error:  %5.3f\n', ser); 
end



