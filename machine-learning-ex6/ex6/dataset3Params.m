function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

Values = [0.01 0.03 0.1 0.3 1 3 10 30]';
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

%set  global (initial) error value as 1 as error is a fraction, so it cannot be more than 1
errorVal = 1;
for i = 1:length(Values)
    for j = 1:length(Values)
        model = svmTrain(X, y, Values(i), @(x1,x2)(gaussianKernel(x1,x2,Values(j))));
        pred = svmPredict(model, Xval);
        error = mean(double(pred ~= yval));
        % check if error for these values of C and sigma is less than global error,
        % set global error to current error if yes
        if error<errorVal
            errorVal = error;
            C = Values(i);
            sigma = Values(j);
        endif;
    endfor;
endfor;    

% =========================================================================

end
