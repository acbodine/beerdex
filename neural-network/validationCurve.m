function [lambda_vec, error_train, error_val] = ...
    validationCurve(labels, hidden_layer_size, ...
        initial_nn_params, Xtrain, ytrain, Xval, yval)

    lambda_vec = [0; 0.0001; 0.001; 0.003; 0.006; 0.009; 0.01; 0.03;];

    error_train = error_val = zeros(size(lambda_vec, 1), 1);

    input_layer_size = size(Xtrain, 2);

    for l = 1:size(lambda_vec, 1)
        % Train theta for training set, and record cost.
        [theta_train] = trainNN(labels, ...
            hidden_layer_size, initial_nn_params, ...
            Xtrain, ytrain, lambda_vec(l));

        % Evaluate training error.
        [cost_train, grad_train] = nnCostFunction(theta_train, ...
            input_layer_size, hidden_layer_size, ...
            size(labels, 2), Xtrain, ytrain, 0);
        error_train(l) = cost_train;

        % Evaluate cross validation error.
        [cost_val, grad_val] = nnCostFunction(theta_train, ...
            input_layer_size, hidden_layer_size, ...
            size(labels, 2), Xval, yval, 0);
        error_val(l) = cost_val;
    endfor
end
