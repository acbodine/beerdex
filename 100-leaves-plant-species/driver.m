setenv("GNUTERM", "qt");

addpath ../neural-network/

% Cleanup
clear; close all; clc;

fprintf('Loading data...\n');

[X, y, labels] = parseRawData('data_Tex_64.txt', 64);
% [X, y, labels] = parseRawData('data_Sha_64.txt', 64);
% [X, y, labels] = parseRawData('data_Mar_64.txt', 64);

% Map labels to numerical values.
str_labels = labels;
labels = (1:size(labels, 1))';

% Establish some helpful variables.
[Xtrain, ytrain, Xval, yval, Xtest, ytest] = partitionFeatures(X, y, labels);
input_layer_size = size(Xtrain, 2);
hidden_layer_size = mean([input_layer_size, size(labels, 1)]);

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, size(labels, 1));

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

lambda = 0.03;
% lambda = 0;

% Skip cross-validation if lambda is already chosen.
if lambda == 0
    fprintf('\nCross-Validating Neural Network... \n')

    lambda_vec = [0; 0.0001; 0.001; 0.003; 0.006; 0.009; 0.01; 0.03;];

    [lambda_vec, error_train, error_val] = validationCurve(lambda_vec, ...
        labels, hidden_layer_size, ...
        initial_nn_params, Xtrain, ytrain, Xval, yval);

    close all;
    plot(lambda_vec, error_train, lambda_vec, error_val);
    legend('Train', 'Cross Validation');
    xlabel('lambda');
    ylabel('Error');

    % TODO: Select optimal lambda.
    smallestErr = min(error_val);
    idx = error_val(:, 1) == smallestErr;
    lambda = lambda_vec(idx, 1);

    fprintf('\nLearned best lambda: %f\n', lambda);

    pause;
endif

fprintf('\nTraining Neural Network... \n')

nn_params = trainNN(labels, hidden_layer_size, initial_nn_params, ...
    Xtrain, ytrain, lambda);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 size(labels, 1), (hidden_layer_size + 1));

% save "learnedContext.mat" Theta1 Theta2 labels Xtrain ytrain Xval yval Xtest ytest lambda hidden_layer_size;

pred = predict(Theta1, Theta2, Xtest);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

rmpath ../neural-network/
