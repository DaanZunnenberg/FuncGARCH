   
function [depth, outpoint]=fbplot(data,x,depth,method,show,prob,color,outliercol,barcol,fullout,factor)
% Produces functional boxplots or enhanced functional boxplots of the given functional data. 
% It can also be used to carry out functional data ordering based on band depth. 
%
%Arguments
%
%	data: a p-by-n functional data matrix where n is the number of curves, and p is defined below.
%         alternatively, a functional data object or functional parameter
%         object can also be specified. 
%	x:	the x coordinates of curves. Defaults to 1:p where p is the number of x coordinates.
%	method: the method to be used to compute band depth. Can be one of "BD2", "MBD" or "Both" 
%	        with a default of "MBD".
%	depth:	a vector giving band depths of curves. If missing, band depth computation is conducted.
%	show: logical. If TRUE (the default) then a functional boxplot is produced. If not, band depth
%	      and outliers are returned.
%	prob: a vector giving the probabilities of central regions in a decreasing order, then an enhanced 
%	      functional boxplot is produced. Defaults to be 0.5 and a functional boxplot is plotted.
%	color:	a vector giving the colors of central regions from light to dark for an enhanced functional 
%	       boxplot. Defaults to be magenta for a functional boxplot.
%	outliercol: color of outlying curves. Defaults to be red.
%	barcol: color of bars in a functional boxplot. Defaults to be blue.
%	fullout: logical for plotting outlying curves. If FALSE (the default) then only the part outside the
%	         box is plotted. If TRUE, complete outling curves are plotted.
%	factor: the constant factor to inflate the middle box and determine fences for outliers. Defaults to 
%	        be 1.5 as in a classical boxplot.
%
%Details
%
%	For functional data, the band depth (BD) or modifed band depth (MBD) allows for ordering a sample of 
%	curves from the center outwards and, thus, introduces a measure to define functional quantiles and 
%	the centrality or outlyingness of an observation. A smaller rank is associated with a more central 
%	position with respect to the sample curves. BD usually provides many ties (curves have the same depth
%       values), but MBD does not. "BD2" uses two curves to determine a band. The method "Both" uses BD2 first 
%	and then uses MBD to break ties. The computation is carried out by the fast algorithm proposed by
%	Sun et al. (2012).
%
%Value
%
%	depth:	band depths of given curves.
%	outpoint: column indices of detected outliers.
%
%Author(s)
%
%	Ying Sun: sunwards@stat.osu.edu
%
%	Marc G. Genton: marc.genton@kaust.edu.sa
%
%References
%
%       Sun, Y., Genton, M. G. and Nychka, D. (2012), "Exact fast computation of band depth for large functional 
%       datasets: How quickly can one million curves be ranked?" Stat, 1, 68-74.
%
%	Sun, Y. and Genton, M. G. (2011), "Functional Boxplots," Journal of Computational and Graphical Statistics,
%	20, 316-334.
%
%	Lopez-Pintado, S. and Romo, J. (2009), "On the concept of depth for functional data," Journal of the American
%	Statistical Association, 104, 718-734.
%

% %%default values
% 
% factor=1.5; 
% fullout='False'; 
% barcol='b'; 
% outliercol='r'; 
% color='m'; 
% prob=0.5; 
% show='True';
% method='MBD'; 
% depth=[]; 

% Examples
%
% 	clear all;
%
% 	ncasem = 39;
% 	ncasef = 54;
% 	nage   = 31;
% 
% 	fid = fopen('hgtm.dat','rt');
% 	hgtmmat = reshape(fscanf(fid,'%f'),[nage,ncasem]);
% 
% 	fid = fopen('hgtf.dat','rt');
% 	hgtfmat = reshape(fscanf(fid,'%f'),[nage,ncasef]);
% 
% 	age = [ 1:0.25:2, 3:8, 8.5:0.5:18 ]';
%
%%fbplot of boys' height
%	fbplot(hgtmmat,age,depth,method,show,prob,color,outliercol,barcol,fullout,factor)
%	xlim([0.5,18.5])
%	ylim([60,200])
%	xlabel('Age (Years)')
%	ylabel('Height (cm)')
%	title('Boys')
%
%%fbplot of girls' height
%	fbplot(hgtfmat,age,depth,method,show,prob,color,outliercol,barcol,fullout,factor)
%	xlim([0.5,18.5])
%	ylim([60,200])
%	xlabel('Age (Years)')
%	ylabel('Height (cm)')
%	title('Girls')
 
function combinat=combinat(n,p)
if n<p 
combinat=0;
else
   combinat=nchoosek(n,p);
end
end

%BD2
function dp=fBD2(data)
	[p,n]=size(data);
	[val,rv]=sort(data');
	[xv, rmat] = sort(rv);
	down=min(rmat')-1;
	up=n-max(rmat');
	dp=(up.*down+n-1)/combinat(n,2);
end
%MBD
function dp=fMBD(data)
    [p,n]=size(data);
	[val,rv]=sort(data');
	[xv, rmat] = sort(rv);
	down=rmat'-1;
	up=n-rmat';
	dp=(sum(up.*down)/p+n-1)/combinat(n,2);
end
%%default values

if nargin<11, 	factor=1.5;  end
if nargin<10, 	fullout='False'; end
if nargin<9, 	barcol='b'; end
if nargin<8, 	outliercol='r'; end
if nargin<7, 	color='m'; end
if nargin<6, 	prob=0.5; end
if nargin<5, 	show='True'; end
if nargin<4, 	method='MBD'; end
if nargin<3, 	depth=[]; end
if nargin<2,    x=[]; end

% If data is an fd object or fdPar object extract it

%if(isa_fdPar(data)), data = getfd(data); end
%if(isa_fd(data))
%   if isempty(x) || length(x)==1,
%      rr = getbasisrange(getbasis(data));
%      if isempty(x), npt = 101;
%     else npt = x; end
%     x = linspace(rr(1),rr(2),npt)'; %  end
%  data = eval_fd(x,data);
%end

% default value for x

[tp,n]=size(data);
if isempty(x), x = (1:tp)'; end
if (length(x) ~= tp), error('Dimensions of data and x do not match'); end


  %compute band depth	
  if isempty(depth)
	if strcmp(method,'BD2')
        depth=fBD2(data);
    elseif strcmp(method,'MBD')
            depth=fMBD(data);
    elseif strcmp(method,'Both')
            depth=round(fBD2(data)*10000)+fMBD(data);  
	end
  end
    
	[dp_s,index]=sort(depth,'descend');
    for pp=1:length(prob)
		m=ceil(n*prob(pp));%at least 50%
		center=data(:,index(1:m));
		out=data(:,index((m+1):n));
		inf=min(center,[],2)';
		sup=max(center,[],2)';
		if prob(pp)==0.5 %check outliers
			dist=factor*(sup-inf);
			upper=sup+dist;
			lower=inf-dist;
			%outlier column
			outly=sum(or(data<=lower'*ones(1,n),data>=upper'*ones(1,n)));
			outpoint=find(outly);
			out=data(:,outpoint);
            good=data;
            good(:,outpoint)=[];
			maxcurve=max(good,[],2)';
			mincurve=min(good,[],2)';
			if sum(outly)>0
				if show 
				plot(x,out,'--r');
				end
			end
			barval=(x(1)+x(tp))/2;
            loc=find(sort([x;barval])==barval);
			bar=loc(1);
			if show
			hold all;
			line([x(bar) x(bar)],[maxcurve(bar) sup(bar)],'Color',barcol,'LineWidth',2);
			hold all;
		    line([x(bar) x(bar)],[mincurve(bar) inf(bar)],'Color',barcol,'LineWidth',2);
			end
		end
		
		if show 
			hold all;
            [xinv,xindex]=sort(x,'descend');
            xx=[x;xinv];
    		supinv=sup(xindex);
            yy=[inf,supinv];
			h=fill(xx,yy,color(pp));
			if prob(pp)==0.5
			set(h,'edgecolor',barcol,'LineWidth',2);
			else 
			set(h,'edgecolor',NA);
			end
		end
		if show
			hold all;
			plot(x,data(:,index(1)),'Color','k','LineWidth',2);
			plot(x,maxcurve,'Color','b','LineWidth',2);
			plot(x,mincurve,'Color','b','LineWidth',2);
			if fullout
				if sum(outly)>0 
					hold all;
					plot(x,out,'Color',outliercol);
				end
			end
		end
    end
    hold off;
depth
outpoint
end
