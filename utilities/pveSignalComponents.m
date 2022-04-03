function [pve_total,pve_true_comp,pve_est_comp] = pveSignalComponents(est_components,true_components)
%pveSignalComponents Computes the proportion of variance explained for the
%                       signal, each of the true components, and each of 
%                       the estimated components
            pve_func=@(x,y) 1-(norm(x-y)/norm(x))^2;
                total_approx=sum(cell2mat(est_components(:)'),2);
                total_test=sum(cell2mat(true_components(:)'),2);
                pve_total=pve_func(total_test,total_approx);
                ntrue=numel(true_components);
                nest=numel(est_components);
                pve_true=zeros(ntrue,nest);
                pve_est=zeros(ntrue,nest);
                for ii=1:ntrue
                    for jj=1:nest
                        pve_true(ii,jj)=pve_func(true_components{ii},est_components{jj});
                        pve_est(ii,jj)=pve_func(est_components{jj},true_components{ii});
                    end
                end
                
                
                pve_true_comp=zeros(ntrue,1);
                for ii=1:ntrue
                    jj=find(pve_true(ii,:)>0);
                    if ~isempty(jj)
                        pve_true_comp=pve_func(true_components{ii},sum(cell2mat(reshape(est_components(jj),1,[])),2));
                    end
                end
                
                pve_est_comp=zeros(nest,1);
                
                for jj=1:nest
                    ii=find(pve_est(:,jj)>0);
                    if ~isempty(ii)
                        pve_est_comp(ii)=pve_func(est_components{jj},sum(cell2mat(reshape(true_components(ii),1,[])),2));
                    end
                end
                

end

