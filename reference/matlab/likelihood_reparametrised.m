function likelihood = construct_likelihood_repara(mY,vb1,dK,n,mBsplinesSparseMat,vtau,vtheta)

T = size(mY,2);
dnu = vtheta(1);     % degree of freedom parameter
ddelta = vtheta(2);  % dependence parameter
vomega = vtheta(3:dK+3);   % level parameter
mB = vtheta(dK+4:2*dK+4);  % scale parameter
mA = vtheta(end-dK:end);   % score paremeter
mLambdaOU_delta = exp(-pdist2(vtau,vtau,'fasteuclidean')/ddelta); % construct covariance matrix


mBsplinesMat = full(mBsplinesSparseMat);
mBsplinesMat = [ones(n,1) mBsplinesMat];

likelihood = 0;
vb_now = vb1;
vy_now = mY(:,1);
Temp1 = (dnu + n)/(2*dnu);
for id = 2:T
    vsigma_now = zeros(n,1);
    for id2 = 1:n
        vsigma_now(id2) = mBsplinesMat(id2,:)*vb_now;
    end
    
    Temp2 =mLambdaOU_delta\(vy_now./exp(vsigma_now/2));
    Temp3 = vy_now.*mBsplinesMat./exp(vsigma_now/2);
    Temp4 = 1+ (vy_now./exp(vsigma_now/2))'*Temp2/dnu;

    likelihood = likelihood + (-0.5*logdet(exp(vsigma_now).*mLambdaOU_delta)...
        - (dnu+n)/2*log(Temp4));

    density_score = -0.5*sum(mBsplinesMat,1)' ...
        + Temp1*Temp4^(-1)*Temp3'*Temp2;
    vb_now = vomega + mB.*vb_now + mA.*density_score;
    vy_now = mY(:,id);

end
likelihood = likelihood + T*(gammaln((dnu+n)/2) -...
    gammaln(dnu/2) - n/2*log(pi*dnu));
end