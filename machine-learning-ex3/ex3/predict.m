function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add bias term to X
X = [ones(m, 1) X];

z2 = X * Theta1'; 
a2 = sigmoid(z2); % 5000 * 25

% add bias term to a2 m is unchanged for X and a2
a2 = [ones(m,1) a2];
%calculate z3 for a3
z3 = a2 * Theta2';

% predict p from a3

a3 = sigmoid(z3);

%a3 contains predictions for each example, we need to return the index of max value in predictions

[m,p] = max(a3,[],2);



% =========================================================================


end
