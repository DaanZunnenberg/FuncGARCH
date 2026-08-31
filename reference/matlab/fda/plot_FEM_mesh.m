function plot_FEM_mesh(pts, tri)

%  Last modified 14 June 2017 by Jim Ramsay

nt = size(tri,1);
phdl = plot(pts(:,1), pts(:,2), 'bo');
set(phdl, 'LineWidth', 2)
hold on
for i = 1:nt
    phdl = plot([pts(tri(i,1),1),pts(tri(i,2),1)], ...
                [pts(tri(i,1),2),pts(tri(i,2),2)], 'b-');
    set(phdl, 'LineWidth', 2)
    phdl = plot([pts(tri(i,2),1),pts(tri(i,3),1)], ...
                [pts(tri(i,2),2),pts(tri(i,3),2)], 'b-');
    set(phdl, 'LineWidth', 2)
    phdl = plot([pts(tri(i,3),1),pts(tri(i,1),1)], ...
                [pts(tri(i,3),2),pts(tri(i,1),2)], 'b-');
    set(phdl, 'LineWidth', 2)
end
hold off
