function [F, grad, Pvec, Pden] = ...
    FEM_density_fit(cvec, basisobj, Xvec, Yvec, K1, lambda)

%FEMDENSITY computes the negative log likelihood and its first and second
%  derivatives with respect to CVEC for the inhomogeneous Poisson
%  model for the density of spatially distributed data whose
%  coordinates are XVEC and YVEC.  Covariate values in Zmat may also be
%  used to model the data
%
%  Input: 
%
%  CVEC      ...  A column vector of coefficients for first order finite
%                 element functions
%  BASISOBJ  ...  An order 1 FEM basis object defined by function
%                 CREATE_FEM_BASIS
%  XVEC      ...  X-coordinates of points of length N
%  YVEC      ...  Y-coordinates of points of length N
%  K1        ...  the stiffness matrix  for penalizing slope
%  LAMBDA    ...  the penalty parameter for penalizing slope
%
%  Output:
%
%  F    ... Objective function value
%  GRAD ... Gradient with respect to coordinates
%  PVEC ... Vector of probabilities
%  PDEN ... Norming constant for probabilities

% Last modified on 27 June 2016 by Jim.

if nargin < 2
    error('Less than two arguments provided.');
end

if nargin < 6,  lambda = 0; end
if nargin < 5,  K1 = [];    end

%  check that basisobj is of FEM type

type = getbasistype(basisobj);
if ~strcmp(type, 'FEM')
    error('BASISOBJ is not of type FEM.');
end

%  retrieve nodeStruct from basisobj and arrays from nodeStruct

nodeStruct = getbasispar(basisobj);
p          = nodeStruct.p;
t          = nodeStruct.t;
t          = t(:,1:3);
nodeindex  = nodeStruct.nodeindex;
ntri       = size(nodeindex,1);
nodes      = nodeStruct.nodes;
if isempty(K1) && lambda > 0  
    K1 = stiff1(nodeStruct);  
end

%  points default to nodes of mesh

if nargin < 4,  Yvec = nodes(:,2);  end
if nargin < 3,  Xvec = nodes(:,1);  end

%  check Xvec

if ~isa(Xvec,'double')
   error('Xvec is not a numerical array')
else
   Xvec  = Xvec(:);     % treat Xvec as a column vector
end

%  check Yvec

if ~isa(Yvec,'double')
   error('Yvec is not a numerical array')
elseif length(Yvec(:))~=length(Xvec)
   error('Yvec is not the same length as Xvec')
else
   Yvec=Yvec(:);     % treat Yvec as a column vector
end

N = length(Xvec);

% identify element containing point in vector (Xvec(i),Yvec(i))
% if no element contains a point, ind(i) is NaN

tricoef = tricoefCal(p, t);
indpts = zeros(N,1);
for i=1:N
   indpts(i) = insideIndex(Xvec(i), Yvec(i), p, t, tricoef);
end

if any(isnan(indpts))
    error('Points are outside of boundary.');
end

%  evaluate basis functions at points

phimatData = eval_FEM_fd(Xvec, Yvec, basisobj);
nbasis = size(phimatData,2);

%  check cvec, which should of length NBASIS if there are no covariates,
%  or NBASIS + Q is there Zmat is N by q

q = 0;
if length(cvec) ~= nbasis+q
    error('CVEC is not of correct length.');
end

%  compute first term of log likelihood

lnintens = phimatData*cvec;
F = -sum(lnintens);
if nargout > 1 
    grad = -sum(phimatData)';       
end
Pnum = exp(lnintens);

%  loop through elements and compute integrals over elements

nquad = 4;
if nargout > 1
    gradsum = zeros(nbasis,1);
end
tripts = zeros(3,2);
int0el = zeros(ntri,1);
for el=1:ntri
    %  point indices for this triangle
    triel = t(el,:);
    %  set up 3 by 2 matrix of vertex coordinates
    for j=1:3
        tripts(j,:) = p(triel(j),:);
    end
    %  set up quadrature points and weights for this triangle
    %  Note 10May14:  pull these out of basis object if present.
    %  Check iMac to see if this is done there.
    [Xel,Yel,Wx,Wy] = triquad(nquad,tripts);
    Xquad      = reshape(Xel,nquad^2,1);
    Yquad      = reshape(Yel,nquad^2,1);
    %  evaluate basis at quadrature points
    phimatquad = eval_FEM_fd(Xquad, Yquad, basisobj);
    %  log intensity at quadrature points
    lnintensel = phimatquad*cvec(1:nbasis);
    %  intensity values
    intensvec  = exp(lnintensel);
    intensmat  = reshape(intensvec,nquad,nquad);
    %  compute approximation to integrated intensity
    int0el(el) = Wx'*intensmat*Wy;
    if nargout > 1
        %  derivatives with respect to coefficients
        gmat   = phimatquad.*repmat(intensvec,1,nbasis);
        garray = reshape(gmat,nquad,nquad,nbasis);
        %  approximation to integrated derivatives
        for k=1:nbasis
            gradsum(k) = gradsum(k) + Wx'*squeeze(garray(:,:,k))*Wy;
        end
    end
end;
int0sum = sum(int0el);
%  compute vector of probabilities at observation points
if nargout > 2
    Pden    = int0sum;
    Pvec = Pnum./Pden;
end

%  complete calculation of negative log likelihood and its gradient

F = F + N*log(int0sum);
if nargout > 1
    grad = grad + N*gradsum./int0sum;
end

%  apply the roughness penalty

if lambda > 0
    cvec0 = cvec(1:nbasis);
    F  = F + lambda.*cvec0'*K1*cvec0;
    if nargout > 1
        grad = grad + 2.*lambda.*K1*cvec0;
    else
        grad = [];
    end
end

