function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
computeCostMulti(X, y, theta);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    h = X * theta;
    hmy = (h-y);
    val = zeros(1,size(X,2));
    for i = 1:size(X,2)
        hmy(:,i:i) = (h-y) .* X(:,i:i);
        %hmyx =(h-y) .* X(:,2);
    
        val(:,i:i) = sum(hmy(:,i)) * (1/m);
        %val1 = sum(hmyx) * (1/m);
    end
    %temp1 = theta(1) - (alpha * val) ;
    %temp2 =  theta(2) - (alpha * val1) ;
    %theta(1) = temp1;
    %theta(2) = temp2;
    
    %delta = [val;val1];
    for j = 1:size(X,2)
        theta(j) = (theta(j) -(val(:,j:j) .* alpha)); 
    end
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
