function [W,b,F,ypre] = MVAR(Xl,Xu,Yl,lambda, s, r,maxIter)
%%  code of "Scalable Multi-View Semi-Supervised Classification via Adaptive Regression"
%%  Tao et al. IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 26, NO. 9, SEPTEMBER 2017
%% input:
%%%%     Xl & Xu: cell, each cell element is a (labeled or unlabeled) data
%%%%              matrix from one view, each row is a data point.
%%%%              the data matrix may be pre-processed by data normalization
%%%%              or centerization when necessary
%%%%     Yl: label matirx of the labeled part, 1-of-c coding, i.e., Y(i,j) = 1, if i belongs to
%%%%        the j-th class, otherwise Y(i,j) = 0.
%%%%     s: array, the weights manually assigned for each points,usually the weights
%%%%        for labeled points is larger than unlabeled points.
%%%%     lambda: array, the trade-off parameters for each view
%%%%     r>1: the parameter for redistribute weights over views
if ~exist('maxIter','var')
    maxIter = 30;
end

s = s(:);
viewNum = length(Xl);
[nlSmp,nClass] = size(Yl);
nuSmp = size(Xu{1},1);
nSmp = nlSmp + nuSmp;
viewDim = zeros(viewNum,1);

%% initialization
alpha = ones(viewNum,1)/viewNum;  %% initialize the view weight
W = cell(viewNum,1); b = W;  
for v = 1:viewNum
    viewDim(v) = size(Xl{v},2);
    [W{v}, b{v}] = least_squares_regression(Xl{v},  Yl,  lambda(v));
end


%% parameters for stopping the loop
h = 6;
do_loop = 1;
obj = [];
stop_crit = 1e-6;
%% b is absorbed into W
X = cell(viewNum,1);
XX = cell(viewNum,1); WW = XX; 
en = ones(nSmp,1);
B= cell(viewNum,1);  bb = B;
for v = 1:viewNum
    X{v} = [Xl{v}; Xu{v}];
    XX{v} = [X{v} en];
    WW{v} = [W{v}; b{v}'];
    B{v} = diag(ones(nSmp,1));
end

%% begin the loop
iter = 1;
F = zeros(nSmp,nClass);
F(1:nlSmp,:) = Yl;
while iter < maxIter && do_loop

    %%%% update F
    Fu = zeros(nuSmp,nClass);
    for v = 1:viewNum
        temp = Xu{v}*W{v} +ones(nuSmp,1)*b{v}';
        Fu = Fu + alpha(v)^r*(B{v}(nlSmp+1:nSmp,nlSmp+1:nSmp)*temp);
    end
    [~,ypre] = max(Fu,[],2);

    Ypre = zeros(nuSmp,nClass);
    for i = 1:nClass
        Ypre(ypre == i,i) = 1;
    end
    F(nlSmp+1:nSmp,:) = Ypre;

    %%%% update matrix B
    for v = 1:viewNum
        Ev = XX{v}*WW{v} - F;
        bb{v} = 0.5*s./sqrt(sum(Ev.*Ev,2) + eps); %%%% eps is added to avoid being divided by 0.
        B{v} = diag(bb{v});
    end
    %%%% update WW
    %%%% when the matrix G is close to singularity, the results may be not
    %%%% precise, this will also affect the convergence of objective functioin
    for v = 1:viewNum
        if viewDim(v) < nSmp
            G = XX{v}'*B{v}*XX{v} + lambda(v)*eye(viewDim(v)+1);
            WW{v} = G\(XX{v}'*B{v}*F);
        else
            G = XX{v}*XX{v}' + lambda(v)*diag(1./bb{v});
            WW{v} = XX{v}'*(G\F);
        end
        W{v} = WW{v}(1:end-1,:);
        b{v} = WW{v}(end,:)';
    end


    %%%% update alpha and calculate objective value
    resErr = zeros(viewNum,1); 
    for v = 1:viewNum
        ErrMat = XX{v}*WW{v} - F;
        resErr(v) = sum(s'*sqrt(sum(ErrMat.*ErrMat,2))) + lambda(v)*sum(sum(WW{v}.*WW{v}));
    end
    if r > 1
        alpha = resErr.^(1/(1-r));
        alpha = alpha/sum(alpha);
    else
        error('r must larger than 1.\n');
    end
    if iter > 1
        obj(iter-1) = sum((alpha.^r).*resErr);
    end
    if iter > h
        temp = obj(end-h+1:end);
        objdiff(iter-h+1) = (max(temp) - min(temp))/max(temp);
    end
    if exist('res','var') && objdiff(end) < stop_crit
        do_loop = 0;
    end
    iter = iter + 1;
            
end

end



function [W, b] = least_squares_regression(X,  Y,  gamma)

% X:                                     each row is a data point
% Y:                                     each row is an target data point: such as  [0, 1, 0, ..., 0]'
% gamma:                                 a positive scalar

[N, dim] = size (X);
[~, dim_reduced] = size(Y);

% first step,  remove the mean!
XMean = mean(X);                                       
XX = X - repmat(XMean, N, 1);                    

W = [];
b = [];
if dim < N
    
    % W = pinv( XX * XX' + gamma * eye(dim)) * (XX * Y');
    %  Note that the above sentence can be repalced by the following sentences. So, it is more fast.
    t0 =  XX' * XX + gamma * eye(dim);
    W = t0 \ (XX' * Y);   
    b = Y -  X*W;    
    b = mean(b)';       
    
else
    t0 = XX * XX' + gamma * eye(N);
    W = XX' * (t0 \ Y);
     b = Y -  X*W;     
     b = mean(b)';       
    
end
end

