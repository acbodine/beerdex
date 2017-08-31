function [Xtrain, ytrain, Xval, yval, Xtest, ytest] = partitionFeatures(X, y, labels)
    % We really want to base our sample selection off of labels. Keep tallys,
    % and partition X according to desired proportions.
    % Desired proportions for the Xtrain, Xval, and Xtest sets
    % respectively according to X: 60%, 20%, and 20%.

    Xtrain = Xval = Xtest = [];
    ytrain = yval = ytest = [];

    c = size(labels, 1);

    tallys = [labels, zeros(c, 1)];
    for i = 1:size(y, 1)
        idx = tallys(:, 1) == y(i);
        tallys(:, 2) = tallys(:, 2) + idx;
    endfor

    for t = 1:size(tallys, 1)
        nTrain = round(6 * tallys(t, 2) / 10);
        nVal = round((tallys(t, 2) - nTrain) / 2);

        % Find indexes in y that match current tally index. Filter class
        % samples from y and X.
        classIdxs = y(:, 1) == tallys(t, 1);
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
        Xtest = [Xtest; Xclass(nTrain+nVal+1:tallys(t, 2), :)];
        ytest = [ytest; yclass(nTrain+nVal+1:tallys(t, 2), :)];
    endfor
end
