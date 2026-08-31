function basismat = fourier_new(evalarg, nbasis, period, nderiv)
%  FOURIER  Computes the NDERIV derivative of the Fourier series basis
%    for NBASIS functions with period PERIOD, these being evaluated
%    at values in vector EVALARG.
%  Returns an N by NBASIS matrix BASISMAT of function values

%  last modified 4 April 2016

evalarg = evalarg(:); %  ensure that EVALARG is a column vector
n       = length(evalarg);
range   = [min(evalarg),max(evalarg)];

%  set default number of basis functions

if nargin < 2
    nbasis = n;
end

%  set default period

if nargin < 3
    period = range(2) - range(1);
end

%  set default order of derivative

if nargin < 4
    nderiv = 0;
end

%  check argument values

if nbasis <= 0,  error('NBASIS not positive');  end
if period <= 0,  error('PERIOD not positive');  end
if nderiv <  0,  error('NDERIV negative');      end

%  make number of basis functions odd if required

if 2*floor(nbasis/2) == nbasis
    nbasis = nbasis + 1;
end

%  set up the basis matrix

basismat = zeros(n,nbasis);

%  set up some constants

omega  = 2*pi/period;
jvec   = 2:2:(nbasis-1);

%  check whether argument sequence is equally spaced

% crit = sqrt(var(diff(evalarg))/var(evalarg));
% if crit > eps
    argmat = (omega .* evalarg) * (jvec./2);
    Smat   = sin(argmat);
    Cmat   = cos(argmat);    
% else
%     argminvec = (omega .* evalarg(1)) * (jvec./2);
%     deltavec  = (omega .* (evalarg(2)-evalarg(1))) * (jvec./2);
%     alphavec  = 2*sin(deltavec./2).^2;
%     betavec   = sin(deltavec);
%     Smat = zeros(n,length(jvec));
%     Cmat = Smat;
%     Smat(1,:) = sin(argminvec);
%     Cmat(1,:) = cos(argminvec);
%     for i=2:n
%         Sincr = alphavec.*Smat(i-1) - betavec.*Cmat(i-1);
%         Cincr = alphavec.*Cmat(i-1) + betavec.*Smat(i-1);
%         Smat(i,:) = Smat(i-1,:) - Sincr;
%         Cmat(i,:) = Cmat(i-1,:) - Cincr;
%     end
% end

if nderiv == 0
    %  The fourier series itself is required.
    basismat(:,1) = 1/sqrt(2);
    basismat(:,jvec)   = Smat;
    basismat(:,jvec+1) = Cmat;
else
    %  A derivative of the fourier series is required.
    onen = ones(n,1);
    basismat(:,1) = 0.0;
    if nderiv == floor(nderiv/2)*2
        mval = nderiv/2;
        fac  = onen * (((-1).^mval).*((jvec./2).*omega).^nderiv);
        basismat(:,jvec)   =  fac .* Smat;
        basismat(:,jvec+1) =  fac .* Cmat ;
    else
        mval  = (nderiv-1)/2;
        fac  = onen * (((-1).^mval).*((jvec./2).*omega).^nderiv);
        basismat(:,jvec)   =  fac .* Cmat;
        basismat(:,jvec+1) = -fac .* Smat;
    end
end

basismat = basismat./sqrt(period./2);

