function [X, y, labels] = parseRawData(filename, n)

    X = dlmread(filename, ',');
    X = X(:, 2:n+1);

    fid = fopen(filename);

    line = fgetl(fid);
    parts = strsplit(line, ",");
    labels = cellstr([parts{1}]);
    y = [1];

    line = fgetl(fid);
    while (line != -1)
        parts = strsplit(line, ",");

        if strcmp(parts{1}, labels) == 0
            labels = cellstr([labels, parts{1}]);
        endif

        % TODO: Make this get index into labels instead of assuming order here.

        y = [y; size(labels, 2)];

        line = fgetl(fid);
    end

    fclose(fid);

    labels = labels';

end
