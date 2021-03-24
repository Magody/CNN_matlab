function zero_pad = zeroPad(X, pad)
    % X: dimension (n_H, n_W, n_C, m), solo se agrega padding a n_H-W
    [n_H, n_W, n_C, m] = size(X);
    new_n_H =  n_H + 2 * pad;
    new_n_W =  n_W + 2 * pad;

    zero_pad = zeros(new_n_H, new_n_W, n_C, m);

    for i = (1+pad):n_H+pad
        for j = (1+pad):n_W+pad
            for k = 1:n_C
                zero_pad(i, j, k, :) = X(i-pad, j-pad, k, :);

            end
        end
    end
end