function atoms= flatAtomsToGroupedAtoms(flat_atoms,group_dim)
%flatAtomsToGroupedAtoms convert an atom matrix to cell array of atom
%matrices given defined by the group dimensions
%assumes the type indices can be sequentially grouped
Nscale=numel(group_dim);
%Scale_dim
atoms=cell(Nscale,1);
cumul_group=cumsum(group_dim(1:end-1));
group_offsets=[0;cumul_group(:)];
group_edges=[0;group_dim(:)+group_offsets];
for ii=1:Nscale
        kk_idx=flat_atoms(:,2)> group_edges(ii)...
               & flat_atoms(:,2)<=group_edges(ii+1);
        atoms{ii}=flat_atoms(kk_idx,:);
        atoms{ii}(:,2)= atoms{ii}(:,2)-group_offsets(ii);
end
end

