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

