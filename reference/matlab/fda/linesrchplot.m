function linesrchplot(x,dx,fx)
T = range(x);
delta = T/10;
n = length(x);
% for i=1:n
%   fprintf('%10.4f %12.6f %12.6f\n', [x(i), dx(i), fx(i)]);
% end
plot(x, fx, 'bo')
hold on
for i=1:n-1
    plot([x(i),x(i)+delta], [fx(i),fx(i)+dx(i)*delta],'b-')
end
plot([x(n),x(n)+delta], [fx(n),fx(n)+dx(n)*delta],'r-')
hold off