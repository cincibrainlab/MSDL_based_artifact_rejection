function atoms_all = multistageMP(signal,grouped_waves,signal_params,model_params)
% Approximates a signal using the multiple passes of matching pursuit
%approx_segment - (signal vector) segment of original signal to approximate
%AA - (cell array) set of subspace filters
%WW - (cell array) set of spatial filters for each subspace
%Approximation_passes - (scalar) number of passes in the non-overlapping approximation algorithm
%Subset_density - (vector) source rate for each filter set
Ngroups=numel(grouped_waves);
coverage=1;
Approximation_passes=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if isfield(signal_params,'block_starts')
        block_starts=signal_params.block_starts;
    else
        block_starts=[];
    end
%Get any optional user parameters
if isfield(model_params,'coverage')
    coverage=model_params.coverage;
end
if numel(coverage)==1
    coverage=coverage*ones(Ngroups,1);
end
if isfield(model_params,'approximation_passes')
    Approximation_passes=model_params.approximation_passes;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%derived parameters
group_length=cellfun(@(x) size(x,1),grouped_waves);
atoms_per_scale=ceil(coverage*length(signal)./group_length);

atoms_all=cell(numel(grouped_waves),1);
V=signal;
for ii=1:numel(grouped_waves)
    t=tic;
    if ~isempty(block_starts)
        mp_func=@(X,D) blockbasedMP(X,D,atoms_per_scale(ii),...
            block_starts,model_params.nonneg_flag);
    else
        mp_func=@(X,D) shiftInvariantMP(X,D,atoms_per_scale(ii),model_params.nonneg_flag);
    end
    [ atoms_all{ii}, ~ ,Ff ] = multipassMP(V,grouped_waves{ii},mp_func,Approximation_passes);
    V=V-Ff;
    %finish-separate
    time_subspace=toc(t);
    %fprintf('Subspace %i of %i: atoms %i, rmse %g, run time %gs\n',ii,numel(grouped_waves),size(atoms_all{ii},1),norm(V),time_subspace);
end

end

