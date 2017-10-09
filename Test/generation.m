function Data = generation(batch_size)
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
Ua = inputs_generation(batch_size);
for i=1:size(Ua,1)
    i
    U = Ua(i, :);
    DEM       = spm_DEM_generate(M,U,{P},{8,16},{16});

    BOLD = full(DEM.pU.v{1}');
    neural = full(DEM.pU.v{2}');
    state = full(DEM.pU.x{1}');

    Data.U(i, :) = U;
    Data.BOLD(:, i, :) = BOLD;
    Data.neural(:, i, :) = neural;
    Data.state(:, i, :) = state;
    Data.N = 64;
end
 U = Data.U;
 BOLD = Data.BOLD;
 neural = Data.neural;
 state = Data.state;
 N = Data.N;

save data.mat U BOLD neural state N
 