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
my = y.*(-1);
oy = 1.-y;
h = X*theta;
first = (1/m)*(my.*log(sigmoid(h)));
sec = (1/m)*oy.*log(1-sigmoid(h));
lterm = (lambda/(2*m)) * (theta.^2);
t= first-sec;
J1 = sum(t);
J2 = sum(lterm)-lterm(1);
J = J1+J2;
grad = (1/m)*(X'*(sigmoid(h)-y));
temp = grad(1);
grad = (1/m)*(X'*(sigmoid(h)-y))+(lambda/m)*theta;
grad(1) = temp;




% =============================================================

end
