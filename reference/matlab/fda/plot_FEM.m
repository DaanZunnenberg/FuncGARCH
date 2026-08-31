function plot_FEM(FEMfd, Xgrid, Ygrid)
% PLOT  Plots a FEM object FEMFD over its triangular mesh.
% The coefficient vector for FEMfd object defines the heights of the
% surface at the vertices.  The vertices are in
% NBASIS by 2 matrix P extracted from the FEM basis object.
%
% Last modified on 7 June 2014

if ~isa_fd(fdobj)
    error('FDOBJ is not an FD object');
end
basisobj = getbasis(fdobj);
if ~strcmp(getbasistype(basisobj), 'FEM')
    error('Basis is not of type FEM.');
end

coefmat = getcoef(fdobj);

params = getbasispars(basisobj);
pts = params.p;

if is_missing(Xgrid)
    xmin  = min(pts(:,1));
    xmax  = max(pts(:,1));
    nx    = 201;
    Xgrid = linspace(xmin, xmax, nx)';
else
    nx    = length(Xgrid);
end

if is_missing(Ygrid)
    ymin = min(pts(:,2));
    ymax = max(pts(:,2));
    ny   = 201;
    Ygrid = linspace(ymin, ymax, ny)';
else
    ny    = length(Ygrid);
end

Xmat = Xgrid * ones(1,ny);
Ymat = ones(nx,1) * Ygrid';
Xmat = zeros(nx,ny);
for numc = 1:nx
    Xmat(:,numc) = Xvec;
end
Ymat = zeros(nx,ny);
for numc = 1:ny
    Ymat(:,numc) = Yvec;
end

evalmat = eval_FEM_fd(Xmat, Ymat, fdobj);

nsurf = size(coefmat,2);
for isurf = 1:nsurf
    evalmati = reshape(evalmat(:,isurf),nx, ny);
    mesh(Xgrid,Ygrid,evalmati)
    % image(Xgrid,Ygrid,evalmati,col=heat.colors(100), xlab='', ylab='', asp=1)
    % contour(Xgrid,Ygrid,evalmati,add=T)
    if nsurf > 1
        pause
    end
end
        
        
        
