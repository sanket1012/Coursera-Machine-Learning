function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
J1=0;
grad = zeros(size(theta));
dim=size(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta=sigmoid(X*theta);

for i=1:m
    J=J+((-y(i,1)*log(h_theta(i,1)))-(1-y(i,1))*log(1-h_theta(i,1)));
end

for j=1:dim(1)
    J1=(lambda/2)*(theta(j,1))^2;
end
J=(1/m)*(J+J1);

i=1;
grad(i,1)=(1/m)*sum((h_theta(1:m,1)-y(1:m,1))'*X(1:m,i));

for i=2:dim(1)
    grad(i,1)=((1/m)*sum((h_theta(1:m,1)-y(1:m,1))'*X(1:m,i)))+(lambda/m)*theta(i,1);
end


% =============================================================

end
