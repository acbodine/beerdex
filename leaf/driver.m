setenv("GNUTERM", "qt");

addpath ../neural-network/

% Cleanup
clear; close all; clc;

fprintf('Loading data...\n');

All_Data = load('leaf.csv');

% Splice All_Data into X and y matrices.
y = All_Data(:, 1);
X = All_Data(:, 3:size(All_Data, 2));

% Scale and normalize feature data.
X = scaleAndNormalizeFeatures(X);

% Find unique class labels from y.
labels = unique(y);

% Map y values to be indexes into labels.
for l = 1:size(labels, 1)
    idx = y(:, 1) == labels(l);
    y(idx, 1) = l;
endfor

% Randomly partition data set into training, cross-validation, and test sets.
[Xtrain, ytrain, Xval, yval, Xtest, ytest] = partitionFeatures(X, y, labels);

% Establish neural network properties
input_layer_size = size(Xtrain, 2);
hidden_layer_size = mean([input_layer_size, size(labels, 1)]);

fprintf('\nInitializing Neural Network Parameters ...\n');

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, size(labels, 1));

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

lambda = 0;

if lambda == 0
    fprintf('\nCross-Validating Neural Network to find best lambda... \n');

    lambda_vec = [0; 0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1;];

    [lambda_vec, error_train, error_val] = validationCurve(lambda_vec, ...
        labels, hidden_layer_size, initial_nn_params, ...
        Xtrain, ytrain, Xval, yval);

    close all;
    plot(lambda_vec, error_train, lambda_vec, error_val);
    legend('Train', 'Cross Validation');
    xlabel('lambda');
    ylabel('Error');

    % Select optimal lambda.
    smallestErr = min(error_val);
    idx = error_val(:, 1) == smallestErr;
    lambda = lambda_vec(idx, 1);

    fprintf('\nLearned best lambda: %f\n', lambda);

    pause;
endif

fprintf('\nTraining Neural Network... \n');

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
