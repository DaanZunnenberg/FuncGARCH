function basisobj = create_fdVar_basis(rangeval, I, J)
%  CREATE_FDVAR_BASIS Creates a functional basis object with type 
%  'fdVar' thatcan be used to define a covariance surface.  This 
%  object can consist of one or more trapezoidal domains defined over a 
%  common range.  The range is defined in argument RANGEVAL, which is also
%  the RANGEVAL field of the basis object.
%  Each trapezoidal region is defined by the number of vertical
%  intervals in the corresponding element in I, and the number of
%  horizontal elements in the corresponding element in J.  I and J
%  must have the same lengths, and their common length is the number
%  of trapezoidal domains defined.
%  The PARAMS field for the basis object is a struct object with fields
%  I and J.  The TYPE field is 'fdVar'.  The NBASIS field is the
%  sum over i of (I(i)+1)*(J(i)+1)
%
%  Arguments:

%  Last modified 20 Octosber 2015 by Jim Ramsay

if nargin < 3,  J = I;  end

%  check I and J

nI = length(I);
nJ = length(J);

if nI ~= nJ
    error('I and J do not have same lengths.');
end

nI = length(I);

if any(I) <= 0
    error('I has zero or negative entries.');
end

if any(J) < 0
    error('I has negative entries.');
end

if any(floor(I) ~= I)
    error('I has non-integer values.');
end

if any(floor(J) ~= J)
    error('J has non-integer values.');
end

%  check RANGEVAL

if length(rangeval) == 1
    if rangeval <= 0
        error('RANGEVAL is a single value that is not positive.');
    end
    rangeval = [0,rangeval];
end

if rangechk(rangeval) ~= 1
    error('RANGEVAL is not a legitimate range.');
end

nbasis = 0;
for i=1:nI
    nbasis = nbasis + (I(i)+1)*(J(i)+1);
end

%  construct basis object

type = 'fdVar';

params.I = I;
params.J = J;

basisobj = basis(type, rangeval, nbasis, params, [], [], {}, {});



end
