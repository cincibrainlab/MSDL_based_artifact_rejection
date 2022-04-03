function [group_waves,training_error_ratio,atoms,model_params ] = multistageWaveformLearning(signal,signal_params,model_params)
%multistageWaveformLearning A multistage shift-invariant dictionary learning algorithm
%--------------
% Inputs:
%--------------
% signal         [signal vector] input
% signal_params  [struct]
%  |
%   -- block_starts [vector] the start indices of non-contiguous blocks
%   -- freq [scalar]  sampling frequency
% model_params   [struct]
%  |
%  (common parameters to all learning methods)
%   -- model_sizes
%   -- coeff_per_filter(:),Filters_per_scale(:)),...
%   -- subsample_rates [vector] downsampling rates for different scales
%   -- model_sizes [matrix, m x 2]  m = number of scales
%                          1st column) timesteps (at subsampled rate) per
%                                       wavefrom at each scale
%                          2nd column) number of waveforms per scale
%   -- coverage  [scalar  [0,1] ] proportion of signal that waveforms
%       should cover in a single approximation pass
%   -- nonneg_flag [bool]  if true then non-negative constraint is imposed
%   -- approximation_passes [scalar] number of passes in the
%       non-overlapping approximation algorithm during learning
%  (specfic parameters)
%   -- minimum_freq  : [scalar  [0,1] ] minimum occurcence rate to keep a waveform
%   -- window_func  : [function handle] windowing function {@(x) tukeywin(x,.1)}
%                                       (argument is window length) returns window vector
%   -- max_correlation  : [scalar  [0,1] ] maximum cross correlation for different waveforms
%--------------
% Outputs:
%--------------
% group_waves - [cell array]   numel(group_waves) = number of scales
%                           Each entry is a matrix of waveforms, columns
%                           corrspond to different waveforms
% training_error_ratio [scalar]  ratio of error norm to signal norm
% atoms [cell array]   numel(atoms) = number of scales
%                           Each entry is a matrix representing the atoms
%                           of the decomposition as stored by the matching
%                           pursuit algorithms  each row is (i,j,a,e) where
%                               i - time index, j - waveform index, a -
%                               coefficient, e - squared error after adding
% model_params [struct] same as in input, but now with defaults set
%-----------------------
% Internal Dependencies:
%-----------------------
% MPSVD
% blockbasedMP
% shiftInvariantMP
% multipassMP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Firstly, check if there are initial waveforms
%if there they exist use them to get the lengths and dimensions
if isfield(model_params,'init_waveforms')
    Ngroups=numel(model_params.init_waveforms);
    group_dim=zeros(1,Ngroups);
    group_length=zeros(1,Ngroups);
    for ii=1:Ngroups
        dict_init=model_params.init_waveforms{ii};
        [group_length(ii),group_dim(ii)]=size(dict_init);
    end
else
    if isfield(model_params,'model_sizes');
        group_length=model_params.model_sizes(:,1);
        group_dim=model_params.model_sizes(:,2);
        Ngroups=size(group_dim,1);
    else
        error('Need either some initial waveforms (init_waveforms) or their sizes (model_sizes)\n');
    end
    for ii=1:Ngroups
        dict_init=randn(model_params.model_sizes(ii,:));
        model_params.init_waveforms{ii}=dict_init;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set-up default parameters
coverage=1;
Do_nonneg=0;
Approximation_passes=4;
subsample_rates=ones(Ngroups,1);
block_starts=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get any optional user parameters
if isfield(signal_params,'block_starts')
    block_starts=signal_params.block_starts;
end
if isfield(model_params,'coverage')
    coverage=model_params.coverage;
end
if numel(coverage)==1
    coverage=coverage*ones(numel(subsample_rates),1);
end
if isfield(model_params,'nonneg_flag')
    Do_nonneg=model_params.nonneg_flag;
end
if isfield(model_params,'approximation_passes')
    Approximation_passes=model_params.approximation_passes;
end
if isfield(model_params,'subsample_rates')
    subsample_rates=model_params.subsample_rates;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set user fields for MPSVD that were passed in (none of which are required)
learn_params=[];
if ~isfield(model_params,'window_func')
    learn_params.window_func=@(x) tukeywin(x,.1);%windowing function
else
    learn_params.window_func=model_params.window_func;
end
if isfield(model_params,'nonneg_flag')
    learn_params.nonneg_flag=model_params.nonneg_flag;
end
if isfield(model_params,'minimum_freq')
    learn_params.minimum_freq=model_params.minimum_freq;
end
if isfield(model_params,'max_correlation')
    learn_params.max_correlation=model_params.max_correlation;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%derived parameters
atoms_per_scale=ceil(coverage*length(signal)./subsample_rates./group_length);

v=signal;%signal to learn from/decompose
%Set-up results
group_waves=cell(Ngroups,1);
atoms=group_waves;
for ii=1:Ngroups
    if subsample_rates(ii)>1
        Vsub=decimate(v,subsample_rates(ii));
    else
        Vsub=v;
    end
    if ~isempty(block_starts)
        mp_func_sub=@(X,D) blockbasedMP(X,D,atoms_per_scale(ii),...
            ceil(block_starts/subsample_rates(ii)),Do_nonneg);
        mp_func=@(X,D) blockbasedMP(X,D,atoms_per_scale(ii),...
            block_starts,Do_nonneg);
    else
        mp_func_sub=@(X,D) shiftInvariantMP(X,D,atoms_per_scale(ii),Do_nonneg);
        mp_func=mp_func_sub;
    end
    if isfield(signal_params,'freq')
        learn_params.freq=signal_params.freq/subsample_rates(ii);
    end
    
    dict_init=model_params.init_waveforms{ii};
    subsampled_waves=MPSVD(Vsub,dict_init,mp_func_sub,learn_params);
    N_waves=size(subsampled_waves,2);
    if subsample_rates(ii)>1%resample
        waves=zeros(size(interp(subsampled_waves(:,1),subsample_rates(ii)),1),N_waves);%back correct storage
        for jj=1:N_waves
            waves(:,jj)=interp(subsampled_waves(:,jj),subsample_rates(ii));
        end
    else
        waves=subsampled_waves;
    end
    waves=bsxfun(@rdivide,waves,sqrt(sum(waves.^2)));
    group_waves{ii}=waves;
    %approximate signal with learned filters and remove the approx
    [atoms{ii},~,approx]=multipassMP(v,group_waves{ii},mp_func,Approximation_passes);
    v=v-approx;
end
training_error_ratio=norm(v)/norm(signal);
end


