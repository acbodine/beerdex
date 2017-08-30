setenv("GNUTERM", "qt");

addpath neural-network/

% Cleanup
clear; close all; clc;

fprintf('Loading data...\n');

addpath 100-leaves-plant-species/

[X, y, labels] = parseRawData('100-leaves-plant-species/data_Tex_64.txt', 64);
% [X, y, labels] = parseRawData('100-leaves-plant-species/data_Sha_64.txt', 64);
% [X, y, labels] = parseRawData('100-leaves-plant-species/data_Mar_64.txt', 64);

rmpath 100-leaves-plant-species/

% Establish some helpful variables.
[Xtrain, ytrain, Xval, yval, Xtest, ytest] = partitionFeatures(X, y, labels);
numClasses = size(labels, 2);
input_layer_size = size(Xtrain, 2);
hidden_layer_size = mean([input_layer_size, numClasses]);

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, numClasses);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% lambda = 0.01;
lambda = 0;

% Skip cross-validation if lambda is already chosen.
if lambda == 0
    fprintf('\nCross-Validating Neural Network... \n')

    [lambda_vec, error_train, error_val] = validationCurve(labels, ...
        hidden_layer_size, initial_nn_params, Xtrain, ytrain, Xval, yval);

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
                 size(labels, 2), (hidden_layer_size + 1));

% save "100-leaves-plant-species/learnedContext.mat" Theta1 Theta2 labels Xtrain ytrain Xval yval Xtest ytest lambda hidden_layer_size;

pred = predict(Theta1, Theta2, Xtest);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);

rmpath neural-network/
