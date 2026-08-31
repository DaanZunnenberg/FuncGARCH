function basismat = getbasismatrix(evalarg, basisobj, nderiv)
%  GETBASISMATRIX   Computes the basis matrix evaluated at arguments in
%    EVALARG associated with basis.fd object BASISOBJ.
%    The returned basis matrix BASISMAT contains the basis
%    derivatives of order NDERIV (0 by default).
%  For each of the available basis types, the arguments required by the
%    evaluation code are set up and then the code is called.

%  last modified 31 March 2016 by Jim Ramsay

if nargin < 3,  nderiv = 0;  end

%  check basisobj

if ~isa_basis(basisobj)
    error('Argument BASISOBJ is not a functional basis object');
end

%  search for stored basis matrix

basismat = getbasisvalues(basisobj, evalarg, nderiv);

if ~isempty(basismat)

    %  stored basis matrix found

    return;

else

    %  evaluate basismatrix

    type     = getbasistype(basisobj);
    nbasis   = getnbasis(basisobj);
    params   = getbasispar(basisobj);
    rangeval = getbasisrange(basisobj);
    dropind  = getdropind(basisobj);
    
    %  for each of these basis types, the arguments required by the
    %  evaluation code are set up and then the code is called.

    switch type
        case 'bspline'
            % 1. B-spline basis
            rangex   = rangeval;
            if isempty(params)
                breaks = rangex;
            else
                breaks   = [rangex(1), params, rangex(2)];
            end
            norder   = nbasis - length(breaks) + 2;
            basismat = bsplineM(evalarg, breaks, norder, nderiv);
        case 'fourier'
            %  2. fourier basis
            period   = params(1);
            basismat = fourier(evalarg, nbasis, period, nderiv);
        case 'monom'
            %  3. monomial basis
            if isstruct(params)
                argtrans  = params.argtrans;
                exponents = params.exponents;
            else
                argtrans  = [0,1];
                exponents = params;
            end
            basismat = monomial(evalarg, exponents, nderiv, argtrans);
        case 'polyg'
            %  4. polygonal basis (also order 2 spline will work, too)
            basismat = polyg(evalarg, params);
        case 'power'
            %  5. power basis (allows non-integer powers)
            basismat = powerbasis(evalarg, params, nderiv);
        case 'expon'
            %  6. exponential basis (one or more rate constants)
            exponents = params;
            basismat = expon(evalarg, exponents, nderiv);
        case 'const'
            %  7. constant basis
            basismat = ones(length(evalarg),1);
        case 'QW'
            %  8. the basis for the modified Weibull quantile function
            basismat = QW(evalarg, nderiv);
        case 'QWM'
            %  9. also the basis for the modified Weibull quantile function
            basismat = QWM(evalarg, nderiv);
        case 'QS'
            %  10. the basis for the modified Weibull quantile function
            %      as a function of surprisal rather than probability
            basismat = QS(evalarg, nbasis, nderiv);
        case 'slide'
            %  11. a series of exponential decays with origins at 
            %  break points
            rangex   = rangeval;
            if isempty(params)
                breaks = rangex;
            else
                breaks = [rangex(1), params(1:(nbasis-1)), rangex(2)];
                rates  = params(nbasis:(2*nbasis-1));
            end
            basismat = slides(evalarg, breaks, rates, nderiv);
        case 'fd'
            %  12.  basis functions that are functional data objects
            basismat = eval_fd(evalarg, params, nderiv);
        case 'FEM'
            nodes    = params.nodes;
            nodemesh = params.nodemesh;
            order    = params.order;
            basismat = FEM(evalarg, nodes, nodemesh, order, nderiv);
        case 'TP'
            error('GETBASISMATRIX not implemented for TP basis objects');
        case 'fdVariance'
            %  13.  basis values for functional covariance surface
            T = max(getbasisrange(basisobj));
            pars = getbasispar(basisobj);
            I = pars.I;
            J = pars.J;
            delta = T./I;
            B = delta.*J;
            basismat = RstCellSetup(evalarg, T, B, delta);
        case 'IRT3PL'
            %  14. basis expansion for psychometric three-parameter
            %  logistic model
            shift    = getbasispar(basisobj);
            basismat = IRT3PL(evalarg, nbasis, shift, nderiv);
        otherwise
            error('Basis type not recognizable');
    end

    %  remove unwanted basis functions if there are any
    
    if ~isempty(dropind)
        index = 1:nbasis;
        for i=1:length(dropind)
            index = index(index ~= dropind(i));
        end
        basismat = basismat(:,index);
    end

end