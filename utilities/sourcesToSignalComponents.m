function [group_approx,group_components ] = sourcesToSignalComponents(grouped_waveforms,grouped_sources )
%sourcesToSignalComponents Creates the components and approximations
%grouped_waveforms - (cell array) set of groups of waveforms
%grouped_sources - (cell array) set of sparse matrices of the source signals

group_approx=cell(numel(grouped_waveforms),1);
group_components=cell(sum(cellfun(@(X) size(X,2),grouped_waveforms)),1);
mm=1;
for ii=1:numel(grouped_waveforms)%across subspaces
for kk=1:size(grouped_waveforms{ii},2)% across filters in subspaces    
%    group_components{mm}=filter(grouped_waveforms{ii}(:,kk),1,full(grouped_sources{ii}(:,kk)));
    [I,~,W]=find(grouped_sources{ii}(:,kk));
    wave_length=size(grouped_waveforms{ii},1);
    sig_length=size(grouped_sources{ii},1);
    I=kron(I,ones(wave_length,1))+kron(ones(numel(I),1),(0:wave_length-1)');    
    W=kron(W,grouped_waveforms{ii}(:,kk));
    W=W(I<=sig_length);
    I=I(I<=sig_length);
    group_components{mm}=sparse(I,ones(numel(I),1),W,sig_length,1);
    
    if kk==1
            group_approx{ii}=group_components{mm};
    else
            group_approx{ii}=group_approx{ii}+group_components{mm}; % ‚â‚Í‚è˜a‚Å•\Œ»‚µ‚Ä‚¢‚é
    end
    mm=mm+1;
end
end

end

