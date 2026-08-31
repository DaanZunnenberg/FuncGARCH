function  sstotal = inprod_TPbasis(basis1,  basis2,  basis3,  basis4,  ...
                                    Lfdobj1, Lfdobj2, Lfdobj3, Lfdobj4, ...
                                    rng, wtfd, EPS, JMAX, JMIN)
%  INPROD_TPBASIS  Computes vectorized four-tensor of inner products of the
%    tensor product of values of respective linear differential operators
%    applied to four bases.  
%    The inner products are approximated by numerical integration using  
%    Romberg integration with the trapezoidal rule. 
%    The inner s can be over a reduced range in RNG and use a 
%    scalar weight function in WTFD.
%
%  Arguments:
%  BASIS1, BASIS2, BASIS3 and BASIS4 ...  these are basis objects,
%            and the inner products of all quadruples of 
%            BASIS1, BASIS2, BASIS3 and BASIS4 are computed.
%  LFDOBJ1, LFDOBJ2, LFDOBJ3 and LFDOBJ4 ... linear differential operators  
%                    for the corresponding basis functions.
%  RNG  ...  Limits of integration
%  WTFD ...  A functional data object defining a weight
%  EPS  ...  A convergence criterion, defaults to 1e-4.
%  JMAX ...  Maximum number of Richardson extrapolation iterations.
%            Defaults to 16.
%  JMIN ...  Minimum number of Richardson extrapolation iterations.
%            Defaults to 5.
%
%  Return:
%  An order NBASIS1*NBASIS2*NBASIS3*NBASIS4 matrix SS of inner products 
%  for each possible pair quadruple of basis functions.

%  last modified 24 March 2015

%  set up default values of arguments

if nargin < 13, JMIN = 5;             end
if nargin < 12, JMAX = 16;            end
if nargin < 11, EPS  = 1E-5;          end
if nargin < 10, wtfd = 0;             end
if nargin <  9, rng  = [];            end
if nargin <  8, Lfdobj4 = int2Lfd(0); end
if nargin <  7, Lfdobj3 = int2Lfd(0); end
if nargin <  6, Lfdobj2 = int2Lfd(0); end
if nargin <  5, Lfdobj1 = int2Lfd(0); end

%  check LFD objects

Lfdobj1 = int2Lfd(Lfdobj1);
Lfdobj2 = int2Lfd(Lfdobj2);
Lfdobj3 = int2Lfd(Lfdobj3);
Lfdobj4 = int2Lfd(Lfdobj4);
  
%  check WTFD

if  isa_fd(wtfd)
    coefw = getcoef(wtfd);
    coefd = size(coefw);
    if coefd(2) > 1
        error('Argument WTFD is not a single function');
    end
end
 
%  check basis function objects

if ~(isa_basis(basis1) && isa_basis(basis2) && ...
     isa_basis(basis3) && isa_basis(basis4))
    error ('The four first arguments are not basis objects.');
end

%  determine NBASIS1 and NASIS2, and check for common range

nbasis1 = getnbasis(basis1) - length(getdropind(basis1));
nbasis2 = getnbasis(basis2) - length(getdropind(basis2));
nbasis3 = getnbasis(basis3) - length(getdropind(basis3));
nbasis4 = getnbasis(basis4) - length(getdropind(basis4));

nprod = nbasis1*nbasis2*nbasis3*nbasis4;

range1  = getbasisrange(basis1);
range2  = getbasisrange(basis2);
range3  = getbasisrange(basis3);
range4  = getbasisrange(basis4);

if nargin < 9 || isempty(rng), rng = range1; end
if rng(1) < range1(1) || rng(2) > range1(2) || ...
   rng(1) < range2(1) || rng(2) > range2(2) || ...
   rng(1) < range3(1) || rng(2) > range3(2) || ...
   rng(1) < range4(1) || rng(2) > range4(2)
   disp(['rng = ',num2str(rng)])
   disp(['range1 = ',num2str(range1)])
   disp(['range2 = ',num2str(range2)])
   disp(['range3 = ',num2str(range3)])
   disp(['range4 = ',num2str(range4)])
      error('Limits of integration are inadmissible.');
end

knotmult = [];

%  check first functional object for knot multiplicities

if strcmp(getbasistype(basis1),'bspline')
    % Look for knot multiplicities in first basis
    params1  = getbasispar(basis1);
    nparams1 = length(params1);
    norder1  = nbasis1 - nparams1;
    knots1   = [range(1)*ones(1,norder1), params1, ...
                range(2)*ones(1,norder1)];
    for i=2:nparams1
        if params1(i) == params1(i-1) || nbasis1 == nparams1 + 1
            knotmult = [knotmult, params1(i)];
        end
    end
end

%  check second functional object for knot multiplicities

if strcmp(getbasistype(basis2),'bspline')
    % Look for knot multiplicities in first basis
    params2  = getbasispar(basis2);
    nparams2 = length(params2);
    norder2  = nbasis2 - nparams2;
    knots2   = [range(1)*ones(1,norder2), params2, ...
                range(2)*ones(1,norder2)];
    for i=2:nparams2
        if params2(i) == params2(i-1) || nbasis2 == nparams2 + 1
            knotmult = [knotmult, params2(i)];
        end
    end
end
   
%  check third functional object for knot multiplicities

if strcmp(getbasistype(basis3),'bspline')
    % Look for knot multiplicities in first basis
    params3  = getbasispar(basis3);
    nparams3 = length(params3);
    norder3  = nbasis3 - nparams3;
    knots3   = [range(1)*ones(1,norder3), params3, ...
                range(2)*ones(1,norder3)];
    for i=2:nparams3
        if params3(i) == params3(i-1) || nbasis3 == nparams3 + 1
            knotmult = [knotmult, params3(i)];
        end
    end
end

%  check fourth functional object for knot multiplicities

if strcmp(getbasistype(basis4),'bspline')
    % Look for knot multiplicities in first basis
    params4  = getbasispar(basis4);
    nparams4 = length(params4);
    norder4  = nbasis4 - nparams4;
    knots4   = [range(1)*ones(1,norder4), params4, ...
                range(2)*ones(1,norder4)];
    for i=2:nparams4
        if params4(i) == params4(i-1) || nbasis4 == nparams4 + 1
            knotmult = [knotmult, params4(i)];
        end
    end
end
   
%  Set up RNGVEC defining subinvervals if there are any
%  knot multiplicities.

rng = getbasisrange(basis1);
if ~isempty(knotmult)
    knotmult = sort(unique(knotmult));
    knotmult = knotmult(knotmult > rng(1) & knotmult < rng(2));
    rngvec = [rng(1), knotmult, rng(2)];
else
    rngvec = rng;
end

%  outer loop is over inter-multiple-knot ntervals

sstotal = zeros(nprod,1);
nrng = length(rngvec);
for irng = 2:nrng
%     disp(['irng = ',num2str(irng)])
    rngi = [rngvec(irng-1),rngvec(irng)];
    %  change range so as to avoid being exactly on
    %  multiple knot values
    if irng > 2
        rngi(1) = rngi(1) + 1e-10;
    end
    if irng < nrng
        rngi(2) = rngi(2) - 1e-10;
    end
    
    %  set up first iteration using only boundary values
    
    width = rngi(2) - rngi(1);
    JMAXP = JMAX + 1;
    h     = ones(JMAXP,1);
    h(2)  = 0.25;
    s     = zeros(JMAXP,nprod);
    if ~isnumeric(wtfd)
        wtvec = eval_fd(wtfd, rngi);
    else
        wtvec = ones(2,1);
    end
    bmat1  = full(eval_basis(rngi, basis1, Lfdobj1));
    bmat2  = full(eval_basis(rngi, basis2, Lfdobj2));
    bmat3  = full(eval_basis(rngi, basis3, Lfdobj3));
    bmat4  = full(eval_basis(rngi, basis4, Lfdobj4));
    tensorprod = Inner_Loop(bmat1, bmat2, bmat3, bmat4, wtvec);
    chs    = width.*tensorprod'./2;
    s(1,:) = chs;
    tnm    = 0.5;
    
    %  now iterate to convergence
    
    for iter = 2:JMAX
%         disp(iter)
        tnm = tnm.*2;
        del = width./tnm;
        x   = rngi(1)+del/2:del:rngi(2);
        nx  = length(x);
        if ~isnumeric(wtfd)
            wtvec = eval_fd(wtfd, x);
        else
            wtvec = ones(nx,1);
        end
        bmat1  = full(eval_basis(x, basis1, Lfdobj1));
        bmat2  = full(eval_basis(x, basis2, Lfdobj2));
        bmat3  = full(eval_basis(x, basis3, Lfdobj3));
        bmat4  = full(eval_basis(x, basis4, Lfdobj4));
        tensorprod = Inner_Loop(bmat1, bmat2, bmat3, bmat4, wtvec);
        chs = width.*tensorprod'./tnm;
        chsold = s(iter-1,:);
        s(iter,:) = (chsold + chs)./2;
        if iter >= 5
            ind   = (iter-4):iter;
            ya    = s(ind,:);
            xa    = h(ind);
            absxa = abs(xa);
            [absxamin, ns] = min(absxa);
            cs = ya;
            ds = ya;
            y  = ya(ns,:);
            ns = ns - 1;
            for m = 1:4
                for i = 1:(5-m)
                    ho      = xa(i);
                    hp      = xa(i+m);
                    w       = (cs(i+1,:) - ds(i,:))./(ho - hp);
                    ds(i,:) = hp.*w;
                    cs(i,:) = ho.*w;
                end
                if 2*ns < 5-m
                    dy = cs(ns+1,:);
                else
                    dy = ds(ns,:);
                    ns = ns - 1;
                end
                y = y + dy;
            end
            ss = y;
            errval = max(abs(dy));
            ssqval = max(abs(ss));
            if all(ssqval > 10*eps)
                crit = errval./ssqval;
                % disp(['errval, ssqval, crit = ', ...
                %       num2str([errval,ssqval,crit])])
            else
                crit = errval;
            end
%             disp(['errval, crit = ',num2str([errval, crit])])
%             plot(1:length(dy),dy,'o')
%             pause
            if crit < EPS && iter >= JMIN
                ss = ss';
%                 disp(['Number of inprod iterations = ',num2str(iter)])
                break
            end
        end
        s(iter+1,:) = s(iter,:);
        h(iter+1)   = 0.25.*h(iter);
    end
    sstotal = sstotal + ss(:);
    if iter == JMAX
        disp(['No convergence after ',num2str(JMAX), ...
            ' steps in INPROD_TPBASIS4.']);
    end
end

