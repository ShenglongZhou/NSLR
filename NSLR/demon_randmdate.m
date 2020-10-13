clc; clear; close all;

p          = 10000; 
n          = ceil(0.2*p);
s          = ceil(0.1*p);
test       = 1;
type       = {'Indipendent','Corrolated'};
[X,y]      = RandomData(n,p,type{test},s,0.5);

out        = NSLR(X,y,s);    
fprintf(' Sparsity:        %d\n', out.sparsity);
fprintf(' CPU time:        %.3fsec\n', out.time);
fprintf(' Logistic Loss:   %5.2e\n', out.logitloss);
fprintf(' Classify Error:  %5.3f\n', out.ser);
fprintf(' Sample size:     %dx%d\n', n,p);

 