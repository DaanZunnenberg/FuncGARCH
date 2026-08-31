function evalmat = eval_FEM_fd(pts, fdobj, nderivs)
% EVAL_FEM_FD evaluates the FEM fd object at points (Xvec,Yvec)
%
% Arguments:
% PTS        ... A two-column matrix with row dimension N.
%                The first column contains the X-coordinates of the points
%                where observations have been made, and the second column
%                contains their Y-coordinates.
% FDOBJ   ... A functional data object whose basis is of the FEM type.
% NDERIVS ... A vector of length 2 containing the orders of derivatives
%             for X and Y, respectively.
% 
%        output:
% EVALMAT   an array of the same size as Xvec and Yvec containing the value  
%           of FELSPLOBJ at (Xvec,Yvec).
%
%  Last modified on 15 June 2017 by Jim ramsay.

%  Set up the arguments if the first argument is a matrix with two
%  columns

%  set default values

if nargin < 3,  nderivs = zeros(1,2);  end

Xvec = pts(:,1);
Yvec = pts(:,2);

%  check the type of FDOBJ

if     isa_fd(fdobj)
    if ~strcmp(getbasistype(getbasis(fdobj)), 'FEM')
        error('The basis object for FDOBJ is not of type FEM.');
    end
elseif isa_basis(fdobj)
    if ~strcmp(getbasistype(fdobj), 'FEM')
        error('The basis object for FDOBJ is not of type FEM.');
    end
else
    error('FDOBJ is neither of FD or BASIS class.');
end

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

%  check derivatives

if length(nderivs) ~= 2
    error('NDERIVS not of length 2.');
end
if sum(nderivs) > 2
    error('Maximum derivative order is greater than two.');
end

N = length(Xvec);

%  get basis

if isa_basis(fdobj)
    basisobj = fdobj;
else
    basisobj = getbasis(fdobj);
end

basismat = eval_FEM_basis(Xvec, Yvec, basisobj, nderivs);

coefs = getcoef(fdobj);

evalmat = full(basismat*coefs);
