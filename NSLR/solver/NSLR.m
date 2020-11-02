function Out = NSLR(X,y,s,pars)

% This code aims at solving the sparse logistic regression with form
%
%    min   L(z) + (lambda/2)\|z\|^2
%    s.t.  \|z\|_0<=s, z\in R^p
%
% where L(z) = sum_i=1^n [ln(1+exp(<z,x_i>))-y_i<z,x_i>]/n
%       X = [x_1 x_2 ... x_n]^T \in R^{n x p}  
%       y = [y_1 y_2 ... y_n]^T \in R^n
%       s is the given sprsity, which is <=n.  
%
% Inputs:
%     X       : Sample data    [x_1 ... x_n]^T \in R^{n x p},       (required)
%     y       : Labels/classes [y_1 ... y_n]^T \in R^n, y_i\in{0,1} (required)
%     s       : Sparsity level, an integer between 1 and p-1,       (required)             
%     pars:     Parameters are all OPTIONAL
%               pars.tau   --  A positive parameter,                (default,15)
%               pars.lam   --  The penalty parameter,               (default,1e-5/n)
%               pars.maxit --  Maximum number of iterations,        (default,2000) 
%               pars.tol   --  Tolerance of the halting condition,  (default,1e-15)
% Outputs:
%     Out.logitloss:     Objective value at Out.sol, i.e., logistic loss
%     Out.sparsity:      Sparsity level of Out.sol
%     Out.normgrad:      L2 norm of the gradient at Out.sol  
%     Out.error:         Error used to terminate this solver 
%     Out.time           CPU time
%     Out.iter:          Number of iterations
%     Out.ser:           Classification error
%     Out.sol:           The sparse solution in \R^p
%
% This code is written by Shenglong Zhou
% and programmed based on the algorithm proposed in 
% Rui Wang, Naihua Xiu, Shenglong Zhou (2020),
% Fast Newton Classification Using Sparse Logistic Regression.
% Send your comments and suggestions to <<< shenglong.zhou@soton.ac.uk >>> 
% Warning: Accuracy may not be guaranteed !!!!! 

t0    = tic;
[n,p] = size(X);

if nargin < 2;   error('Inputs are not enough!');                  end
if nargin < 3;   pars = [];     s = ceil(0.01*p);                  end
if nargin < 4;   pars = [];                                        end
if isfield(pars,'tau');  tau   = pars.tau;   else; tau   = 15;     end
if isfield(pars,'maxit');maxit = pars.maxit; else; maxit = 500;    end
if isfield(pars,'tol');  tol   = pars.tol;   else; tol   = 1e-15;  end  
if isfield(pars,'lam');  lam   = pars.lam;   else; lam   = 1e-5/n; end  

scale  = n;
z      = zeros(p,1); 
eXz    = ones(n,1)/2;
ey     = 1-y;
lam    = lam*(n/scale);
g      = ((0.5-y)'*X)'/scale; 

I      = 1:p;
T0     = [];
mark   = 1;
itmark = -10;
flag   = 0;
obj    = Inf;
Error  = Inf; 

is_pcg = ~(s>=2000 | n>10000);
OBJ    = zeros(maxit+1,1);

fprintf('--------------------------------------\n');
fprintf('Iter        Error         LogisticLoss\n');
fprintf('--------------------------------------\n');    

for iter  = 0 : maxit
    
    if  iter 
        OBJ(iter)=obj; 
        Error = scale/sqrt(p)*sqrt(norm(gT)^2+norm(z(TTc))^2);
        fprintf('%3d        %7.2e         %7.2e\n',iter,Error,obj);
        stop  = (iter>4 && std(OBJ(iter-4:iter)/OBJ(iter))<1e-5);      
        if Error<tol || stop, break; end 
    end   
    
    if ~flag
        if s  < 1e4
        [~,T] = maxk(abs(z-tau*g),s); 
        else
        [~,P] = sort(abs(z-tau*g),'descend');
        T     = P(1:s); 
        end
        T     = sort(T);
        XT    = X(:,T); 
    end   
        gT    = g(T);
        D     = eXz.*(1-eXz)/scale;  
    if  is_pcg
        H     = (D.*XT)'*XT + lam*speye(s); 
    else
        H     = @(x)( lam*x+( (D.*(XT*x))'*XT )' );
    end

    if  flag || isempty(setdiff(T,T0))  
        if is_pcg
        d     =  H\gT;  
        else
        [d,~] = pcg(H,gT,1e-8*s,20);  
        end
    else     
        Tc    = setdiff(I,T);
        TTc   = intersect(T0,Tc); 
        rhs   = gT-(( D.*( X(:,TTc)*z(TTc) ))'*XT)'; 
        if is_pcg
        d     = H\rhs;  
        else
        [d,~] = pcg(H,rhs,1e-8*s,20);  
        end
        
    end
    
    % Amijio line search
        dg     = -sum(d.*gT);
    if ~flag
        dg     = dg-sum(z(TTc).*g(TTc)); 
    end
        obj0   = obj;
        alpha  = 1;    
    for j      = 1:5
        zT     = z(T) - alpha*d;
        Xz     = XT*zT;
        obj    = LogitLoss(Xz,y);     
        if obj < obj0 + alpha*1e-6*dg; break; end        
        alpha  = alpha/2;
    end

    T0    = T;
    z     = zeros(p,1);
    z(T)  = zT;
     
    if abs(obj-obj0)<1e-6*(1+obj0) && Error  < 1e-6 
        lam   = max(1e-8/scale,lam/10);     % reduce the regularized parameter eps
    elseif obj> obj0*10 && iter-itmark > 10 % restart with a lager regularized parameter lam
        lam   = mark*1e-3/scale;
        z     = zeros(p,1);
        Xz    = zeros(n,1);
        tau   = max(3,tau);    
        itmark= iter; 
        mark  = mark+1;
    end
   
    eXz   = 1./(1+exp(Xz));  
    g     = ((ey-eXz)'*X)'/scale + lam*z;
    flag  = (tau < min(abs(zT))/max(abs(g(Tc))));    

    if ~flag && iter && mod(iter,10)==0 && Error>1/iter^2 
       tau = 0.75*tau;    
    end
 
   
end

fprintf('--------------------------------------\n');
Out.sparsity  = nnz(T);
Out.logitloss = obj;
Out.normgrad  = (scale/sqrt(p))*norm(g);
Out.error     = Error;
Out.time      = toc(t0);
Out.iter      = iter;
Out.ser       = nnz(y-max(0,sign(Xz)))/n; 
Out.sol       = z;
end

%logistic loss-------------------------------------------------------------
function obj = LogitLoss(Xz,y)
    if  sum(exp(Xz))==Inf
        Tp  = find(Xz>0);
        Tn  = setdiff(1:length(y),Tp);
        obj = sum(log(1+exp(Xz(Tn)))-y(Tn).*Xz(Tn))+ ...
              sum(Xz(Tp)-y(Tp).*Xz(Tp)); 
    else
        obj = sum(log(1+exp(Xz))-y.*Xz); 
    end
    obj     = obj/length(y);
end
