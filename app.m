addpath(genpath('cnn_utils'));
X = rand(3, 3, 2);

% a = zeroPad(X, 2);

% TEST CONV FORWARD
m = 10; n_H_prev = 5; n_W_prev = 7; 
n_C_prev = 4; n_C = 8; f = 3;

A_prev = ones(n_H_prev, n_W_prev, n_C_prev, m);
W = ones(f, f, n_C_prev, n_C);
b = ones(1, 1, 1, n_C);

A_prev(4, 6, 3, 3) = 4449;
W(3, 2, 4, 5) = 1234;
b(1, 1, 1, 5) = -13;

hparameters.pad = 1;
hparameters.stride = 2;

[Z, cache_conv] = convForward(A_prev, W, b, hparameters);

fprintf("test conv forward: %i\n", sum(sum(sum(sum(Z))))/(120*8) == 273.6);

% TEST POOLING FORWARD

m = 2; n_H_prev = 5; n_W_prev = 5; 
n_C_prev = 3; f = 3; stride = 1;
A_prev = ones(n_H_prev, n_W_prev, n_C_prev, m);
A_prev(3, 4, 2, 1) = 123;
A_prev(1, 1, 3, 2) = -12;


hparameters.f = f;
hparameters.stride = stride;

[A, cache_pool] = poolForward(A_prev, hparameters, 1);
fprintf("test pool max forward: %i\n", sum(sum(sum(sum(A)))) == 786.0);

[A, cache_pool] = poolForward(A_prev, hparameters, 2);
fprintf("test pool average forward: %i\n", ceil(sum(sum(sum(sum(A))))) == 134);

