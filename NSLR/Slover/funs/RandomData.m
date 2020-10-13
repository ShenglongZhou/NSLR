function [X,y,z,out] = RandomData(n,p,type,s,rho)

switch type
    case 'Indipendent'
        I0    = randperm(n);
        y     = ones(n,1);
        I0    = I0(1:n/2); 
        y(I0) = 0;
        X     = repmat(y.*rand(n,1),1,p)+ randn(n,p);
        z     = [];
        out   = [];
    case 'Indipendent-z'       
        I0    = randperm(p);
        z     = zeros(p,1);
        I     = I0(1:s); 
        z(I)  = randn(s,1);
        
        X     = randn(n,p);  
        Xz    = X(:,I)*z(I);
        q     = 1./(1+exp(-Xz));
        y     = zeros(n,1);
        for i = 1:n    
        y(i) = randsrc(1,1,[0 1; 1-q(i) q(i)]);
        end
        
        out.f = logistic_fun(Xz,y);
        out.ser = sum(abs(y-max(0,sign(Xz))))/n;
   
    case 'Corrolated'
        I0    = randperm(p);
        z     = zeros(p,1);
        I     = I0(1:s); 
        z(I)  = randn(s,1);

        v     = randn(n,p);
        X     = zeros(n,p); 
        X(:,1)= randn(n,1);

        for j=1:p-1
        X(:,j+1)=rho*X(:,j)+sqrt(1-rho^2)*v(:,j);
        end

        %y     = binornd(ones(n,1),1./(1+exp(-Xtz)));
        Xz    = X(:,I)*z(I);
        q     = 1./(1+exp(-Xz));
        
        y     = zeros(n,1);
        for i = 1:n    
        y(i)  = randsrc(1,1,[0 1; 1-q(i) q(i)]);
        end
        
        out.f   = LogisticLoss(Xz,y);
        out.ser = sum(abs(y-max(0,sign(Xz))))/n;

end   
