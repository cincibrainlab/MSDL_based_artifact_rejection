function [group_waves,training_error_ratio,atoms,model_params ] = multistageICASelection(signal,signal_params,model_params)
%multistageICASelection A multistage single-channel ICA dictionary learning algorithm
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
%   -- window_func  : [function handle] windowing function {@(x) tukeywin(x,.1)}
%                                       (argument is window length) returns window vector
%   -- max_ica_iterations  : [scalar]  maximum number of iterations for FastICA
%   -- fraction_to_estimate  : [scalar  [0,1] ] proportion of possible
%                               waveforms to learn (e.g., when this parameter
%                               is 0.4 and learning 100-long waveforms yileds 40
%                               waveforms)
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
% External Dependencies:
%-----------------------
% fastica    (After download, http://research.ics.aalto.fi/ica/fastica/ add to path)
%-----------------------
% Internal Dependencies:
%-----------------------
% OMP
% blockbasedMP
% shiftInvariantMP
% multipassMP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set-up default parameters
coverage=1;
Do_nonneg=0;
Approximation_passes=4;
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
%Set user fields for ICA
if ~isfield(model_params,'window_func')
    Win=@(x) tukeywin(x,.1);%windowing function
else
    Win=model_params.window_func;
end
if isfield(model_params,'max_ica_iterations')
    max_ica_iterations=model_params.max_ica_iterations;
else
    max_ica_iterations=100;
end
if isfield(model_params,'fraction_to_estimate')
    fraction_to_estimate=model_params.fraction_to_estimate;
else
    fraction_to_estimate=.4;
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
        %        sub_block_starts=1+floor((block_starts-1)/subsample_rates(ii));
        sub_block_starts=ceil(signal_params.block_starts/subsample_rates(ii));
    else
        Vsub=v;
        sub_block_starts=block_starts;
    end
    if ~isempty(block_starts)
        mp_func=@(X,D) blockbasedMP(X,D,atoms_per_filter(ii),block_starts,Do_nonneg);
        mp_func_sub=@(X,D) blockbasedMP(X,D,atoms_per_filter(ii),sub_block_starts,Do_nonneg);
    else
        mp_func=@(X,D) shiftInvariantMP(X,D,atoms_per_filter(ii),Do_nonneg);
        mp_func_sub=mp_func;
    end
    
    
    N=group_sizes(ii,1);
    Nkeep=group_sizes(ii,2);
    ica_n_estimate=ceil(fraction_to_estimate*N);
    %begin-form Toeplitz matrix of the signal
    Y=repmat(Vsub,1,N);
    tts=[1,sub_block_starts(:)',size(Vsub,1)];
    row_idx=1;
    for block_ii=1:numel(sub_block_starts)+1
        tt=tts(block_ii):tts(block_ii+1);
        for ti=1:N
            Y(row_idx+(0:numel(tt)-1),ti)=cat(1,zeros(ti-1,1),Vsub(tt(1:end-ti+1)));
        end
        row_idx=row_idx+numel(tt);
    end
    Y=Y(1:row_idx-1,:);
    %finish-form
    Y=bsxfun(@times,Win(N)',Y);%window to discourage filters to be at edges
    [ A0, ~] = fastica(Y','approach','symm','g','tanh','numOfIC',ica_n_estimate,'maxNumIterations',max_ica_iterations);
    subsampled_waves=flipud(bsxfun(@rdivide,A0,sqrt(sum(A0.^2))));% flip to make filter order, normalize
    %finish-dictionary construction
    if Do_nonneg==1
        subsampled_waves=cat(2,subsampled_waves,-subsampled_waves);
    end
    %begin-dictonary subselection
    %stage 1 approximate the signal using each dictonary element seperately
    %number of atoms per each assumes uniform distribution with desired number of filters
    %This could be improved by varying the number of atoms used by each
    %waveform
    tic
    N_waves=size(subsampled_waves,2);
    approx=zeros(size(Vsub,1),N_waves);
    for ci=1:N_waves
        [~,~,approx(:,ci)]=mp_func_sub(Vsub,subsampled_waves(:,ci));
    end
    toc
    %stage 2 using orthogonal matching pursuit (OMP) using the components
    % to select a non-redundant set of dictionary elements
    tic
    idx=OMP(Vsub,approx,Nkeep);
    toc
    subsampled_waves=subsampled_waves(:,idx);
    N_waves=size(subsampled_waves,1);
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




