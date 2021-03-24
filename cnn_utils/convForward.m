function [Z, cache] = convForward(A_prev, W, b, hparameters)
%{
Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function

%}


[n_H_prev, n_W_prev, n_C_prev, m] = size(A_prev);
[f, f, n_C_prev, n_C] = size(W);

stride = hparameters.stride;
pad = hparameters.pad;

n_H = floor(n_H_prev + 2*pad - f)/stride + 1;
n_W = floor(n_W_prev + 2*pad - f)/stride + 1;

Z = zeros(n_H, n_W, n_C, m);
A_prev_pad = zeroPad(A_prev, pad);

for sample = 1:m
    a_prev_pad = A_prev_pad(:, :, :, sample);
    
    for i = 0:n_H-1
        
        vert_start = stride * i + 1;
        vert_end = vert_start + f - 1;
       
        for j = 0:n_W-1
            
            horiz_start = stride * j + 1;
            horiz_end = horiz_start + f - 1;
            
            for c = 1:n_C
                a_slice_prev = A_prev_pad(vert_start:vert_end, ...
                    horiz_start:horiz_end, :, sample);
                weights = W(:, :, :, c);
                biases = b(:, :, :, c);
                
                s = a_slice_prev .* weights;
                Zeta = sum(sum(sum(s)));
                Z(i+1, j+1, c, sample) = Zeta + biases;
                
                
                    
            end
                        
        end
    end
    
end

cache.A_prev = A_prev;
cache.W = W;
cache.b = b;
cache.hparameters = hparameters;

end

