function a = distributeValue(dz, n_H, n_W)
%{
Distributes the input value in the matrix of dimension shape
    
Arguments:
dz -- input scalar
shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

Returns:
a -- Array of size (n_H, n_W) for which we distributed the value of dz
    

%}
average = dz/(n_H * n_W);
a = ones(n_H, n_W) * average;

end

