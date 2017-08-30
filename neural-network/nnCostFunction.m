function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% Part 1

a1 = [ones(m, 1) X];

% Add bias units to a1 [5000 x 400] for input layer.
% Apply Theta1 [25 x 401] to a1 [5000 x 401].
z2 = a1 * Theta1';

% Compute activations for hidden layer on z2 [5000 x 25].
a2 = sigmoid(z2);

% Add bias units to a2 [5000 x 25] for hidden layer.
a2 = [ones(size(a2, 1), 1) a2];

% Apply Theta2 [10 x 26] to a2 [5000 x 26] for output layer.
z3 = a2 * Theta2';

% Compute activations for output layer z3 [5000 x 10].
a3 = sigmoid(z3);

H = a3;

% Convert y vector to be a [5000 x 10] matrix that represents
% the output values of y so they are compatible with our 10 output nodes
% in the network.
Y = zeros(size(H));
for i = 1:size(y)(1)
    Y(i, y(i)) = 1;
end;

% Calculate the errors between the hypothesis and expected outputs.
errors = -Y .* log(H) - (1 - Y) .* log(1 - H);

% Calculate the non-regularized cost.
J = (1 / m) * sum(sum(errors, 2));

% Calculate regularization term for the cost.
t1 = Theta1(:, [2 : size(Theta1, 2)]);
t2 = Theta2(:, [2 : size(Theta2, 2)]);
sumTheta1 = sum(sum(t1 .^ 2, 2));
sumTheta2 = sum(sum(t2 .^ 2, 2));
regTerm = (lambda / (2 * m)) * (sumTheta1 + sumTheta2);

J = J + regTerm;

% Part 2

% Implement the backpropagation algorithm.
sigma3 = H - Y;
sigma2 = (sigma3 * Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]))(:, [2:end]);
delta1 = sigma2' * a1;
delta2 = sigma3' * a2;

Theta1_grad = delta1./m;
Theta2_grad = delta2./m;

Theta1_grad = Theta1_grad + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = Theta2_grad + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
