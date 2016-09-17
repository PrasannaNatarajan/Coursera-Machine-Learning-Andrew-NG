function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
%g = (1/(1+exp(-z)));
for i=drange(1:size(z,1))
   for j=drange(1:size(z,2))
       g(i,j)=((1/(1+exp(-z(i,j)))));
   end
end




% =============================================================

end
