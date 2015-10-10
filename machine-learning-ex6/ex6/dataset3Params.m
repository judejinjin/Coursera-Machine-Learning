function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_range = [0.01,0.03,0.1,0.3,1,3,10,30];

sigma_range =[0.01,0.03,0.1,0.3,1,3,10,30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
errors = zeros(size(C_range,2), size(sigma_range,2));

smallest_error = 1e10;

for i = 1:size(C_range, 2)
    for j = 1:size(sigma_range, 2)
        C = C_range(1,i);
        sigma = sigma_range(1,j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        
        predictions = svmPredict(model, Xval);
        %fprintf('%f %f : %f', C, sigma, mean(double(predictions ~= yval)));
        errors(i, j) = mean(double(predictions ~= yval));
        if mean(double(predictions ~= yval)) < smallest_error 
            C_best = C;
            sigma_best = sigma;
            smallest_error = mean(double(predictions ~= yval));
        end 
    end
end


C = C_best;
sigma = sigma_best;



% =========================================================================

end
