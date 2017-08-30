function [Xtrain, ytrain, Xval, yval, Xtest, ytest] = partitionFeatures(X, y, labels)
    % We really want to base our sample selection off of labels. Keep tallys,
    % and partition X according to desired proportions.
    % Desired proportions for the Xtrain, Xval, and Xtest sets
    % respectively according to X: 60%, 20%, and 20%.

    Xtrain = Xval = Xtest = [];
    ytrain = yval = ytest = [];

    c = size(labels, 2);

    tallys = zeros(size(labels, 2), 1);
    for i = 1:size(y, 1)
        tallys(y(i)) = tallys(y(i)) + 1;
    endfor

    for t = 1:size(tallys, 1)
        nTrain = round(6 * tallys(t) / 10);
        nVal = round((tallys(t) - nTrain) / 2);

        % Find indexes in y that match current tally index. Filter class
        % samples from y and X.
        classIdxs = y(:, 1) == t;
        yclass = y(classIdxs, :);
        Xclass = X(classIdxs, :);

        % Randomize class matrixes.
        yclass = yclass(randperm(size(yclass, 1)), :);
        Xclass = Xclass(randperm(size(Xclass, 1)), :);

        % Append to Xtrain, Xval, and Xtest sets based on nTrain and nVal
        % from Xclass.
        Xtrain = [Xtrain; Xclass(1:nTrain, :)];
        ytrain = [ytrain; yclass(1:nTrain, :)];
        Xval = [Xval; Xclass(nTrain+1:nTrain+nVal, :)];
        yval = [yval; yclass(nTrain+1:nTrain+nVal, :)];
        Xtest = [Xtest; Xclass(nTrain+nVal+1:tallys(t), :)];
        ytest = [ytest; yclass(nTrain+nVal+1:tallys(t), :)];
    endfor
end
