function y = softmax(x)

[classes, samples] = size(x);

ex = exp(x);
y = zeros(classes, samples);

for sample = 1:samples
    data_sample = ex(:, sample);
    y(:, sample) = data_sample/sum(data_sample);
end

end

