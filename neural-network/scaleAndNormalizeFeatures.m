function [Scaled] = scaleAndNormalizeFeatures(X)

    [m n] = size(X);

    Scaled = X;

    for j = 1:n
        % Mean value for feature j.
        m_j = mean(X(:, j));

        % Standard deviation of values for feature j.
        s_j = std(X(:, j));

        % Scale and normalize feature values.
        Scaled(:, j) = (X(:, j) .- m_j) ./ s_j;
    endfor

end
