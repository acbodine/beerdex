function [nn_params] = ...
    trainNN(labels, hidden_layer_size, initial_nn_params, X, y, lambda)

    %  After you have completed the assignment, change the MaxIter to a larger
    %  value to see how more training helps.
    options = optimset('MaxIter', 1000);

    num_labels = size(labels, 1);

    input_layer_size = size(X, 2);

    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
                            input_layer_size, hidden_layer_size, ...
                            num_labels, X, y, lambda);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
end
