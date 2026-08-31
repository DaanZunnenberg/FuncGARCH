function plot_FEM_fd(pts, fdobj)
% PLOT  Plots a FEM object FDOBJ over a rectangular grid defined by
% vectors Xgrid and Ygrid

%  Last modified 14 July 2017 by Jim Ramsay.

if ~isa_fd(fdobj)
    error('FDOBJ is not an FD object');
end

coefmat = getcoef(fdobj);

basisobj = getbasis(fdobj);

if ~strcmp(getbasistype(basisobj), 'FEM')
    error('Basis is not of type FEM.');
end

params = getbasispars(basisobj);
tri = params.t;
tri = tri(:,1:3);
basismat = eval_FEM_basis(pts(:,1), pts(:,2), basisobj);
evalmat  = full(basismat*coefmat);

nsurf = size(coefmat,2);
for isurf = 1:nsurf
    evalveci = evalmat(:,isurf);
    trisurf(tri,pts(:,1),pts(:,2),evalveci)
    if nsurf > 1
        pause
    end
end
        
