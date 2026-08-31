clear; clc
set(findobj(0,'type','figure'),'visible','on')
close all


%==================== Import data ====================%
filename = "SP500.csv";
SP500 = importdata(filename);
SP500 = SP500.textdata;
SP500date = datetime(SP500(1,2:end));
SP500time = datetime(SP500(2:end,1),'Format','HH:mm');
SP500idx =  readtable(filename,'NumHeaderLines',1,'TreatAsMissing','NaN');
SP500idx = SP500idx{:,2:end};
SP500 = [];

% Create 20 minute resolution closing prices
tau = 20;
SP500idxTemp = SP500idx(1:tau:size(SP500idx,1),:);
vyTauTemp = diff(SP500idxTemp);
vyTau = vyTauTemp(:,all(~isnan(vyTauTemp)));

%====================================================%


%==================== Estimation ====================%
% Note: The directory of the fdaM function class must be added to path. 
dK = 7;
Bsplinebasis = create_bspline_basis([0, 1], dK, 4);  % Create 7 B-spline basis functions of order 4. 
                                                     % It gives 3 control interior knots.

% B-spline basis functions: full matrix
vtau = 0:1/(size(vyTau,1)-1):1; vtau = vtau';
n = length(vtau);
mBsplinesSparseMat = eval_basis(vtau, Bsplinebasis);

%---------- FunGAS: update principal component scores ----------%
% parameter specifications
vb0 = ones(dK+1,1);             % initival values of principal component scores
vtheta0 = [2.1; 0.001; ones(dK+1,1); -0.5*ones(dK+1,1); 0.1*ones(dK+1,1)];   % initial values of parameters
LB = [1.05; 0.00001; -5*ones(dK+1,1); -2*ones(dK+1,1) ; -0.9*ones(dK+1,1)];  % upper bound for parameters
UB = [50; 1; 15*ones(dK+1,1); 2*ones(dK+1,1) ; 0.9*ones(dK+1,1)];            % upper bound for parameters

% optimization
% vthetaHat contains (in order):
% 1. degree of freedom estiamte;
% 2. dependence estimate;
% 3. omega vector: (dK+1) x 1
% 4. diagonal elements of mB: (dK+1) x 1
% 5. diagonal elements of mA: (dK+1) x 1
fGAS_likelihood = @(vtheta) -construct_likelihood_repara(vyTau,vb0,dK,n,mBsplinesSparseMat,vtau,vtheta);
options = optimoptions('fmincon','MaxFunctionEvaluations',1E4);
vthetaHat = fmincon(fGAS_likelihood,vtheta0,[],[],[],[],LB,UB,[],options);


% plot volatility surface
dnuHat = vthetaHat(1);     % degree of freedom parameter
ddeltaHat = vthetaHat(2);  % dependence parameter
vomegaHat = vthetaHat(3:dK+3);   % level parameter
mBHat = vthetaHat(dK+4:2*dK+4);  % scale parameter
mAHat = vthetaHat(end-dK:end);   % score paremeter
mLambdaOU_deltaHat = exp(-pdist2(vtau,vtau,'fasteuclidean')/ddeltaHat); % construct covariance matrix

vb_now = vb0;
vy_now = vyTau(:,1);
Temp1 = (dnuHat + n)/(2*dnuHat);
mVolatilityHat = zeros(size(vyTau));
T = size(vyTau,2);
mBsplinesMat = full(mBsplinesSparseMat);
vsigma_now = zeros(n,1);
for id = 1:n
    vsigma_now(id) = [1 mBsplinesMat(id,:)]*vb_now;
end
mBsplinesMat = [ones(n,1) mBsplinesMat];
for id = 2:T

    vsigma_now = zeros(n,1);
    for id2 = 1:n
        vsigma_now(id2) = mBsplinesMat(id2,:)*vb_now;
    end

    mVolatilityHat(:,id) = vsigma_now;

    Temp2 =mLambdaOU_deltaHat\(vy_now./exp(vsigma_now/2));
    Temp3 = vy_now.*mBsplinesMat./exp(vsigma_now/2);
    Temp4 = 1+ (vy_now./exp(vsigma_now/2))'*Temp2/dnuHat;

    density_scoreHat = -0.5*sum(mBsplinesMat,1)' ...
        + Temp1*Temp4^(-1)*Temp3'*Temp2;
    vb_now = vomegaHat + mBHat.*vb_now + mAHat.*density_scoreHat;
    vy_now = vyTau(:,id);

end
mVolatilityHat = exp(mVolatilityHat/2);
mVolatilityHat = mVolatilityHat(:,5:end);

vyTaucut = vyTau(:,5:end);
figure (1)
surf(1:size(mVolatilityHat,2),1:size(mVolatilityHat,1),mVolatilityHat,'FaceAlpha',0.5)
grid minor
xlabel('day','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
title('Volatility surface','FontSize',40,'Interpreter','latex')
ax = gca;
ax.FontSize = 30;
xaxisproperties= get(ax, 'XAxis');
xaxisproperties.TickLabelInterpreter = 'latex';
yaxisproperties= get(ax, 'YAxis');
yaxisproperties.TickLabelInterpreter = 'latex';
set(gcf,'Position',[0,0,900,600])
saveas(gcf,'volatility_surface.png')
%====================================================%













