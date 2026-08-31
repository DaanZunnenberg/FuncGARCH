addpath ('../..')

load daily

addpath('/Users/jamesramsay/Documents/MATLAB/SpatialDA/emst')

%  display temperature-precipitation loops

daytempmat = eval_fd(daytime, daytempfd);
dayprecmat = eval_fd(daytime, dayprecfd);

%  compute distance matrix

tempprecdist = zeros(35);

for i=2:35
    for j=1:i-1
        dist = 0;
        for k=1:365
            dist = dist + (daytempmat(k,i) - daytempmat(k,j))^2 ...
                        + (dayprecmat(k,i) - dayprecmat(k,j))^2;
%             dist = dist + (daytempmat(k,i) - daytempmat(k,j))^2;
%             dist = dist + (dayprecmat(k,i) - dayprecmat(k,j))^2;
        end
        tempprecdist(i,j) = sqrt(dist);
        tempprecdist(j,i) = tempprecdist(i,j);
    end
end

%  multidimensional scaling of distance matrix

dim = 3;
[y, stress, fit] = mdscale(tempprecdist,dim);

%  plot points with labels

figure(1)
plot3(y(:,1), y(:,2), y(:,3), 'o')
hold on
for i=1:35
    text(y(i,1)+10, y(i,2), y(i,3), place(i,:))
end
hold off

%  set up a movie for 3D plot

axis([-500,500,-500,500,-50,50])  %  fix axes 

axis vis3d off  %  fix axes for 3d, turn off axes, ticks & etc

%  loop through frames

nframe = 205;
for iframe=1:nframe
    view(-37.5+3*(iframe-1),30) %  change viewpoint
    frame(iframe) = getframe;
end




