function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    
    % The second part of the vectorization is to transpose (X * theta) - y) which gives you a 1*n matrix 
    % which when multiplied by X (an n*2 matrix) will basically aggregate both (h(x)-y)x0 and (h(x)-y)x1. By 
    % definition both thetas are done at the same time. This results in a 1*2 matrix of my new theta's which I 
    % just transpose again to flip around the vector to be the same dimensions as the theta vector. I can 
    % then do a simple scalar multiplication by alpha and vector subtraction with theta.
    
    %     predictions = X*theta;
    %     errors = (predictions-y)';
    %     mul = X(:,1)*errors;
    %     mul = (alpha/m)*mul;
    %     theta(1)=theta(1)-mul;
    %     predictions = X*theta;
    %     errors = X(:,2)*(predictions-y)';
    %     mul = (alpha/m)*mul;
    %     theta(2)=theta(2)-mul;
    
    theta = theta -((1/m) * ((X * theta) - y)' * X)' * alpha;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
J_history;
end
