function hmat = eval_mon(evalarg, Wfdobj, Lfdobj, betam)
%  Evaluates a monotone functional object, 
%    or one of its derivatives. 
%  The monotone functional data object h  is = the form
%           h(x) = (D^{-1} exp Wfdobj)(x)
%  where  D^{-1} means taking the indefinite integral.
%  Note that the linear differential operator object LFDOBJ
%  MUST be an integer in the range 0 to 3.
%  Note that the first two arguments may be interchanged.
%
%  Arguments:
%  EVALARG ... A vector of values at which all functions are to 
%              evaluated.
%  WFDOBJ  ... Functional data object.  
%  LFDOBJ  ... A linear differential operator object
%              applied to the functions that are evaluated.
%
%  Returns:  An array of function values corresponding to the evaluation
%              arguments in EVALARG

%  Last Modified 27 July 2017

if nargin < 2
    error('Number of arguments is less than 2.');
end

%  check LFDOBJ 

if nargin < 3,  Lfdobj = int2Lfd(0);  end

Lfdobj = int2Lfd(Lfdobj);

nderiv = getnderiv(Lfdobj);

%  Exchange the first two arguments if the first is an FD object
%    and the second numeric

if isnumeric(Wfdobj) && isa_fd(evalarg)
    temp    = Wfdobj;
    Wfdobj  = evalarg;
    evalarg = temp;
end

%  check EVALARG

sizeevalarg = size(evalarg);
if sizeevalarg(1) > 1 && sizeevalarg(2) > 1
    error('Argument EVALARG is not a vector.');
end
evalarg = evalarg(:);

%  check WFDOBJ

if ~isa_fd(Wfdobj)
    error('Argument FD is not a functional data object.');
end

coef   = getcoef(Wfdobj);
coefd  = size(coef);
ndim   = length(coefd);
ncurve = coefd(2);
if ndim == 2
    nvar = 1;
    if nargin < 4
        betam = zeros(2,ncurve);
        betam(2,:) = 1;
    else
        betamsize = size(betam);
        if betamsize(1) ~= 2 || betamsize(2) ~= ncurve
            error('Dimensions of BETAM are incorrect.');
        end
    end
else 
    nvar = coefd(3);
    if nargin < 4
        betam = zeros(2,ncurve,nvar);
        betam(2,:,:) = 1;
    else
        betamsize = size(betam);
        if betamsize(1) ~= 2 || betamsize(2) ~= ncurve || ...
           betamsize(3) ~= nvar
            error('Dimensions of BETAM are incorrect.');
        end
    end
end
n = length(evalarg);

if ndim == 2
    hmat = zeros(n, ncurve);
else
    hmat = zeros(n, ncurve, nvar);
end
if nderiv >= 2
    Dwmat = getbasismatrix(evalarg, getbasis(Wfdobj),1);
end
if nderiv == 3
    D2wmat = getbasismatris(evalarg, getbasis(Wfdobj),2);
end

for ivar=1:nvar
    for icurve=1:ncurve
        if nderiv == 0
            if ndim == 2
                hmat(:,icurve) = betam(1,icurve) + ...
                betam(2,icurve)*monfn(evalarg, Wfdobj(icurve));
            else
                hmat(:,icurve,ivar) = betam(1,icurve,ivar) + ...
                betam(2,icurve,ivar)*monfn(evalarg, Wfdobj(icurve,ivar));
            end
        end

        if nderiv == 1
            if ndim == 2
                hmat(:,icurve) = ...
                    betam(2,icurve)* ...
                    exp(eval_fd(evalarg, Wfdobj(icurve)));
            else
                hmat(:,icurve,ivar) = ...
                    betam(2,icurve,ivar)* ...
                    exp(eval_fd(evalarg, Wfdobj(icurve,ivar)));
            end
        end


        if nderiv == 2
            if ndim == 2
                hmat(:,icurve,ivar) = ...
                    betam(2,icurve)*(Dwmat*coef(:,icurve)).*      ...
                    exp(eval_fd(evalarg, Wfdobj(icurve)));
            else
                hmat(:,icurve,ivar) = ...
                    betam(2,icurve,ivar)*(Dwmat*coef(:,icurve,ivar)).* ...
                    exp(eval_fd(evalarg, Wfdobj(icurve,ivar)));
            end
        end

        if nderiv == 3
            if ndim == 2
                hmat(:,icurve,ivar) = ...
                    betam(2,icurve)*((D2wmat*coef(:,icurve)) +      ...
                    (Dwmat*coef(:,icurve))^2).*      ...
                    exp(eval_fd(evalarg, Wfdobj(icurve)));
            else
                hmat(:,icurve,ivar) = ...
                    betam(2,icurve,ivar)*((D2wmat*coef(:,icurve,ivar)) +...
                    (Dwmat*coef(:,icurve,ivar))^2).* ...
                    exp(eval_fd(evalarg, Wfdobj(icurve,ivar)));
            end
        end

        if nderiv > 3
            error('Derivatives higher than 3 are not implemented.');
        end
    end
end

if ndim == 2, hmat = squeeze(hmat);  end

