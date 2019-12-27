%%%% demo_MVAR
load data
viewNum = length(M);
nSmp = size(M{1},1);
r = 2;
maxIter = 20;
%% data can be pre-processed by centerization or normalization if necessary
for v = 1:viewNum
    Xl{v} = M{v}(labelIdx,:);
    Xu{v} = M{v}(unlabelIdx,:);
end
%% 
Yl = Y(labelIdx,:);
s = ones(nSmp,1);
s(1:length(labelIdx)) = 1e4;
lambda = 1e3*ones(viewNum,1);  %%% diffent view can have different lambda value

[W,b,F,ypre] = MVAR(Xl,Xu,Yl,lambda, s, r);

%% accuracy on unlabeled data
[~,gnd] = max(Y,[],2);
gndu = gnd(unlabelIdx);
acc = mean(gndu == ypre);

%% make prediction on unseen data if exists 
% Ft = zeros(nuSmp,nClass);
% for v = 1:viewNum
%     Ft = Ft + alpha(v)^r*(Xt{v}*W{v} +ones(nuSmp,1)*b{v}');
% end
% [~,ypre] = max(Ft,[],2);