function [group_waves,training_error_ratio,atoms,model_params ] = multistageGaborSelection(signal,signal_params,model_params)
%multistageGaborSelection A multistage single-channel dictionary selection algorithm
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
%   -- Ncandidate [scalar] {100} number of candidate filters for the dictionary
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
% blockbasedMP
% shiftInvariantMP
% multipassMP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set-up default parameters
coverage=1;
Do_nonneg=0;
Approximation_passes=4;
freq=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%get frequency if given
if isfield(signal_params,'freq')
 freq=signal_params.freq;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%get required parameters
if isfield(model_params,'model_sizes');
    group_sizes=model_params.model_sizes;
    Ngroups=size(group_sizes,1);
else
    error('Need sizes of waveforms (model_sizes)\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get any optional user parameters
    if isfield(signal_params,'block_starts')
        block_starts=signal_params.block_starts;
    else
        block_starts=[];
    end
if isfield(model_params,'coverage')
    coverage=model_params.coverage;
end
if numel(coverage)==1
    coverage=coverage*ones(Ngroups,1);
end
if isfield(model_params,'nonneg_flag')
    Do_nonneg=model_params.nonneg_flag;
end
if isfield(model_params,'approximation_passes')
    Approximation_passes=model_params.approximation_passes;
end
if isfield(model_params,'subsample_rates')
    subsample_rates=model_params.subsample_rates;
else
    subsample_rates=ones(Ngroups,1);
end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set user fields for Gabor
if isfield(model_params,'Ncandidate')
    Ncandidate=model_params.Ncandidate;
else
    Ncandidate=100;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%derived parameters
atoms_per_filter=ceil(coverage*length(signal)./subsample_rates./prod(group_sizes,2));
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
        mp_func=@(X,D) blockbasedMP(X,D,atoms_per_filter(ii),...
            block_starts,Do_nonneg);
        mp_func_sub=@(X,D) blockbasedMP(X,D,atoms_per_filter(ii),...
            ceil(block_starts/subsample_rates(ii)),Do_nonneg);        
    else
        mp_func=@(X,D) shiftInvariantMP(X,D,atoms_per_filter(ii),Do_nonneg);
        mp_func_sub=mp_func;
    end
    N=group_sizes(ii,1);
    Nkeep=group_sizes(ii,2);
    %begin-Gabor dictionary construction
    %make the Gabor filter templates
    phases=[0 pi/2 pi -pi/2];
    multiplier=1/numel(phases);
    keep_atoms=0;
    while sum(keep_atoms)*numel(phases)<Ncandidate
        n_freq=floor(sqrt(4*Ncandidate/3*multiplier));
        n_bandwidth=ceil(Ncandidate*multiplier/n_freq);
        TE=kron(ones(1,n_freq),logspace(log10(5),log10(N),n_bandwidth));%20*15
        FC=kron(logspace(log10(1),log10(freq/2),n_freq),ones(1,n_bandwidth));%20*15
        keep_atoms=(1./FC)<TE/freq;
        multiplier=multiplier*1.1;
    end
    TE=TE(keep_atoms);
    FC=FC(keep_atoms);
    subsampled_waves=[];
    for phase_ii=phases;
        gf=cos(phase_ii+2*pi*bsxfun(@times,FC,(0:N-1)')/freq);
        gf=gf.*exp(-bsxfun(@rdivide,linspace(-N/2,N/2,N)'.^2,2*TE.^2));
        gf=gf/norm(gf);
        subsampled_waves=cat(2,subsampled_waves,gf);
    end
    
    %finish-dictionary construction
    %begin-dictonary subselection
    %stage 1 approximate the signal using each dictonary element seperately
    %number of atoms per each assumes uniform distribution with desired number of filters
    tic
    N_waves=size(subsampled_waves,2);
    approx=zeros(size(Vsub,1),N_waves); 
    for ci=1:N_waves
        [~,~,approx(:,ci)]=mp_func_sub(Vsub,subsampled_waves(:,ci));
    end
    toc
    %stage 2 using orthogonal matching pursuit using the components
    % to select a non-redundant set of dictionary elements
    tic
    idx=OMP(Vsub,approx,Nkeep);
    toc
    subsampled_waves=subsampled_waves(:,idx);    
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




