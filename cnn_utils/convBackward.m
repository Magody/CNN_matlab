function [dA_prev, dW, db] = convBackward(dZ, cache)
%{
Implement the backward propagation for a convolution function
    
Arguments:
dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
cache -- cache of values needed for the conv_backward(), output of conv_forward()

Returns:
dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
           numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
dW -- gradient of the cost with respect to the weights of the conv layer (W)
      numpy array of shape (f, f, n_C_prev, n_C)
db -- gradient of the cost with respect to the biases of the conv layer (b)
      numpy array of shape (1, 1, 1, n_C)
%}
A_prev = cache.A_prev;
W = cache.W;
b = cache.b;
hparameters = cache.hparameters;

[n_H_prev, n_W_prev, n_C_prev, m] = size(A_prev);
[f, f, n_C_prev, n_C] = size(W);
stride = hparameters.stride;
pad = hparameters.pad;

[n_H, n_W, n_C, m] = size(dZ);

dA_prev = zeros(n_H_prev, n_W_prev, n_C_prev, m);                         
dW = zeros(f, f, n_C_prev, n_C);
db = zeros(1, 1, 1, n_C); 

A_prev_pad = zeroPad(A_prev, pad);
dA_prev_pad = zeroPad(dA_prev, pad);

for sample = 1:m
    a_prev_pad = A_prev_pad(:, :, :, sample);
    da_prev_pad = dA_prev_pad(:, :, :, sample);
    
    for i = 0:n_H-1
        
        
       
        for j = 0:n_W-1
            
            for c = 1:n_C
                
                vert_start = stride * i + 1;
                vert_end = vert_start + f - 1;
                horiz_start = stride * j + 1;
                horiz_end = horiz_start + f - 1;
            
                a_slice = a_prev_pad(vert_start:vert_end, ...
                    horiz_start:horiz_end, :);
                
                
                % Update gradients for the window and the filter's parameters using the code formulas given above
                da_prev_pad(vert_start:vert_end, horiz_start:horiz_end, :) = ...
                    da_prev_pad(vert_start:vert_end, horiz_start:horiz_end, :) + ...
                    W(:,:,:,c) * dZ(i+1, j+1, c, sample);
                dW(:,:,:,c) = dW(:,:,:,c) + (a_slice * dZ(i+1, j+1, c, sample));
                db(:,:,:,c) = db(:,:,:,c) + (dZ(i+1, j+1, c, sample));
                
                    
            end
                        
        end
    end
    [ht, wt, ct] = size(da_prev_pad);
    dA_prev(:, :, :, sample) = da_prev_pad(pad+1:ht-pad, pad+1:wt-pad, :);
end


end

