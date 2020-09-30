function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Unregularized cost
J = sum((X*theta-y).^2)/(2*m);

% Unregularized gradient
h = X*theta;
errors_vector = h - y;
grad = (X'*errors_vector)/(m);

% Regularized cost
theta(1) = 0;
cost_reg = (sum(theta.^2))*(lambda/(2*m));
J = J + cost_reg;

% Regularized gradient
grad_reg = theta*(lambda/m);
grad = grad + grad_reg;

% =========================================================================

grad = grad(:);

end
