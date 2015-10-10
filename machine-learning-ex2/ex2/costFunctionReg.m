function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly

reg_theta = theta .^ 2;
reg_theta(1,1) = 0;

J = (log(sigmoid(X*theta))'*(-1*y) - (1-y)'* log(1-sigmoid(X*theta)))/m + lambda/(2*m) *sum(reg_theta);

reg_theta = lambda/m * theta;
reg_theta(1,1) = 0;
grad = (sigmoid(X*theta)-y)'*X/m + reg_theta';


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
