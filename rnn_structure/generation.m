function Data = generation(inputs)
% set options
%--------------------------------------------------------------------------
M(1).E.linear = 0;                          % linear model
M(1).E.s      = 1;                          % smoothness
 
% level 1
%--------------------------------------------------------------------------
ip      = [1 2 5]';
pE      = spm_hdm_priors(1,5);              % parameters
np      = length(pE);
pC      = sparse(ip,ip,exp(-3),np,np);
pE(6)   = 0.02;
pE(7)   = 0.5;
 
M(1).n  = 4;
M(1).f  = 'spm_fx_hdm';
M(1).g  = 'spm_gx_hdm';
M(1).pE = pE;                               % prior expectation
M(1).pC = pC;                               % prior expectation
M(1).xP = exp(4);                           % prior expectation
M(1).V  = exp(8);                           % error precision
M(1).W  = exp(12);                          % error precision
M(1).ip = ip;
 
% level 2
%--------------------------------------------------------------------------
M(2).l  = 1;                                % inputs
M(2).V  = exp(0);                           % with shrinkage priors
 
% true parameters
%--------------------------------------------------------------------------
P       = M(1).pE;
P(ip)   = P(ip) - P(ip)/8;
 
 
% generate data
%==========================================================================
N         = 64;                             % length of data sequence
for i=1:size(inputs,1)
    input = reshape(inputs(i, :, :), size(inputs,2), size(inputs, 3));
    inp         =zeros(1, N);
    inp([input(1, :)]) = input(2, :);
    U         = exp(-([1:11] - 6).^2/(2.^2))/8; % this is the Gaussian cause
    U         = conv(U,inp);
    U         = U(1:N);

    DEM       = spm_DEM_generate(M,U,{P},{8,16},{16});

    BOLD = full(DEM.pU.v{1}');
    neural = full(DEM.pU.v{2}');
    state = full(DEM.pU.x{1}');

    Data.BOLD(:, i, :) = BOLD;
    Data.neural(:, i, :) = neural;
    Data.state(:, i, :) = state;
    Data.N = N;
end
 