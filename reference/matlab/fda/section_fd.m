function fdsecobj = section_fd(fdobj, secrng, fdParobj)
%  SECTION_FD sets up a functional data object defined over a subsection
%  of the original interval.
%
%  Arguments:
%  FDOBJ    ... Functional data object to be sectioned.
%  SECRNG   ... Subsection range
%  FDPAROBJ ... fdPar object defining the re-smoothing process.
%
%  Returns:
%  FDSECOBJ ... Functional data object defined over the subsection

%  Last modified 26 March 2016

%  extract the basis object from fdobj

basisobj = getbasis(getfd(fdParobj));
nbasis   = getnbasis(basisobj);

%  check that secrng is a subsection of the full range

fullrng  = getbasisrange(getbasis(fdobj));
if secrng(1) < fullrng(1)-1e-10 || secrng(2) > fullrng(2)+1e-10
    disp([secrng(1)-fullrng(1),secrng(2)-fullrng(2)])
    error('RNG not contained within range of FDOBJ.');
end

%  re-smooth the data over this subsection using the smoothing parameters
%  fdParobj

n = max(101, 10*nbasis+1);
tvec = linspace(secrng(1),secrng(2),n)';
fmat = eval_fd(tvec, fdobj);
fdsecobj = smooth_basis(tvec, fmat, fdParobj);

%  set up the output FDSECOBJ

fdnames  = getnames(fdobj);
fdsecobj = putnames(fdsecobj, fdnames);