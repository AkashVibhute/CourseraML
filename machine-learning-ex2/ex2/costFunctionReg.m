function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%calculate hypothesis function for cost function
htheta = sigmoid(X*theta);

%we do not have to penalize theta(1) so reg_theta is a vector from 2nd element of theta to end
reg_theta = theta([2:1:end],:);
J = -(1/m) * (y' * log(htheta) + (1-y') * log(1 - htheta)) + (lambda/(2*m))*sum(reg_theta.^2);


% for gradient calc vectorization won't work? we don't have to regularize theta(1) //theta.0

%calculate gradient for theta(1)
%set gradient same as gradient in logistic regression without regularization
grad_theta =  theta;
grad_theta(1) = 0;
%set gradient same as gradient in logistic regression without regularization
grad = (1/m)*(X' * (htheta - y))+(lambda/m) * grad_theta;
% =============================================================

end
