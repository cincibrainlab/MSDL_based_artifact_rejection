function marked_pp= atomsToSources( atoms,Scale_dim,signal_length,Nkeep )
%atomsToSources convert an atom matrix or a cell array of atom matrices to
%sparse source matrix or matrices

if ~iscell(atoms)
    flag_return_mat=1;
atoms=atoms{1};
else
    flag_return_mat=0;
end
Nscale=numel(atoms);
%Scale_dim
marked_pp=cell(1,Nscale);
for ii=1:Nscale
    amarked_pp=zeros(Nkeep*Scale_dim(ii),3);
    for kk=1:Scale_dim(ii)
        kk_idx=atoms{ii}(:,2)==kk;
        amp=atoms{ii}(kk_idx,3);
        time_idx=atoms{ii}(kk_idx,1);
        filter_idx=atoms{ii}(kk_idx,2);
        [~,ia]=sort(abs(amp),'descend');
        Nkeepkk=min(Nkeep,numel(amp));
        amarked_pp(1+(kk-1)*Nkeep:(kk-1)*Nkeep+Nkeepkk,:)=...
            [time_idx(ia(1:Nkeepkk)),filter_idx(ia(1:Nkeepkk)),amp(ia(1:Nkeepkk))];
    end
    amarked_pp=amarked_pp(amarked_pp(:,1)>0,:);
    marked_pp{ii}=sparse(amarked_pp(:,1),amarked_pp(:,2),amarked_pp(:,3),signal_length,Scale_dim(ii));
end
if flag_return_mat
    marked_pp=cell2mat(marked_pp);
end
end

