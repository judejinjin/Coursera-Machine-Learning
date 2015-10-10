Function [J, grad] = CostFunction(theta, X, y, lambda)

m = length(y); % number of training examples 

H = sigmoid(X*theta); 
T = y.*log(H) + (1 - y).*log(1 - H); 
J = -1/m*sum(T) + lambda/(2*m)*sum(theta(2:end).^2); 

ta = [0; theta(2:end)]; 

grad = (X'*  bsxfun(@minus, H, y))'/m + lambda/m*ta; 

End