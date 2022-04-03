function [per_scale_recon,per_filter_recon] = atomsToSignalComponents(waveforms,atoms,time_window )
%atomsToSignalComponents Uses shift-invariant waveforms and atoms to make signal and components
%with a window of interest

warning('Code uses filter(wave,1,full(source)) which is not efficient for sparse source!\n');
per_scale_recon=0;
per_filter_recon=[];
for kk=1:size(waveforms,2)
    kk_idx=atoms(:,2)==kk & atoms(:,1)<=time_window(2) & atoms(:,1)>=time_window(1);
    t_idx=atoms(kk_idx,1)-time_window(1)+1;
    amps=atoms(kk_idx,3);
    s_kk=sparse(t_idx,1,amps,range(time_window)+1,1);
    y=filter(waveforms(:,kk),1,full(s_kk));% not efficient for very sparse sources!
    per_filter_recon=cat(2,per_filter_recon,y);
    per_scale_recon=per_scale_recon+y;
end

end

