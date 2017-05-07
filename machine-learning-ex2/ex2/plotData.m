function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.

%Find indices of Positive and Negative Examples
pos = find(y==1);
neg = find(y==0);
% Find returns a column vector of the indices where the value of Y is 1/0.
plot(X(pos, 1), X(pos,2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg,2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
% X(pos, 1) are values of Exam 1 Score corresponding to Y=1.
% X(pos, 2) are values of Exam 2 Score corresponding to Y=1.
% X(neg, 1) are values of Exam 1 Score corresponding to Y=0.
% X(neg, 2) are values of Exam 1 Score corresponding to Y=0.









% =========================================================================



hold off;

end
