function Ua = inputs_generation(batch_size)
N = 64;
for i=1:batch_size
    stimulus_times = floor(rand*3+3);
    for j=1:stimulus_times
        input(1, j) = floor(rand*60+3);
        input(2, j) = rand*0.5+0.25;
    end
    
    inp         =zeros(1, N);
    inp([input(1, :)]) = input(2, :);
    U         = exp(-([1:11] - 6).^2/(2.^2))/8; % this is the Gaussian cause
    U         = conv(U,inp);
    Ua(i, 1:N)         = U(1:N);
end
