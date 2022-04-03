function [ dict,approx ] = MPSVD( X,dict,mp_method,params)
% Learn a set of waveforms by alternating matching pursuit, SVD updates
% This uses the semi-nonnegative rank-1 waveform update
% Also checks for redundant waveforms, and those below a minimum rate
%--------------
% Inputs: 
%--------------
% X         [vector] input signal
% dict      [matrix] initial shift-invariant dictionary
% mp_method [function] matching pursuit (2 input arguments: X and Dict, 3 output arguments: atoms,residual,approx)
% params    [struct]
% |
%  -- window_function - [function handle] Univariate function 
%                               (argument is window length) returns window vector
%  -- freq - [scalar] sampling frequency
%  -- minimum_freq [scalar] minimum occurence rate to maintain a dictionary element
%  -- max_correlation [scalar] maximum correlation (<=1) to be considered redundant
%  -- nonneg_flag [bool]  if true then non-negative constraint is imposed
%--------------
% Outputs: 
%--------------
% dict      [matrix] optimized shift-invariant dictionary
% approx    [vector] approximation of input using dictionary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set-up default parameters
maxiter=50;% Max iterations in the inner loop, the updates are not guaranteed to converge
Win=ones(size(dict,1),1);
freq=1;
rate_thresh=1/length(X);
redundant_threshold=.05;
Do_nonneg=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Get any user parameters
if isfield(params,'nonneg_flag')
    Do_nonneg=params.nonneg_flag;
end
if isfield(params,'window_function')
    Win=params.window_function(size(dict,1));
end
if  isfield(params,'freq')
    freq=params.freq;
    rate_thresh=freq/length(X);
end
if isfield(params,'minimum_freq') && isfield(params,'freq')
    rate_thresh=params.minimum_freq;
end
if isfield(params,'max_correlation')
    redundant_threshold=1-params.max_correlation;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=size(dict,1);%waveform length
L=length(X);%signal length
dict=bsxfun(@rdivide,dict,sqrt(sum(dict.^2)));%unit norm waveforms
Of=X(:);
outer_count=1;
while true
    K=size(dict,2);
    Dold=0*dict;
    iii=0;
    while iii<maxiter && mean(sqrt(sum((dict-Dold).^2)))>10^-6
        iii=iii+1;
        Dold=dict;
        [atoms, Rf ,approx] = mp_method(Of,dict);
        for kk=randperm(K)
            kk_idx=(atoms(:,2)==kk)&(atoms(:,1)-1+N<L);
            atom_idx=find(kk_idx);
            if sum(kk_idx)>0
                kk_time_idx=bsxfun(@plus,(0:N-1),atoms(kk_idx,1))';
                F=dict(:,kk)*atoms(kk_idx,3)'+Rf(kk_time_idx);
                XX=bsxfun(@times,Win,F);
                %Check to see if any of the windowed segments are closer to
                %an all-zero vector than to the approximation
                %remove them if it is the case
                d2=sqrt(sum(F.^2));
                d1=sqrt(sum(Rf(kk_time_idx).^2));
                if ~any(d1<d2)
                elseif ~all(d1<d2)
                    idx=d1<d2;
                    XX(:,~idx)=0;
                    Rf(kk_time_idx(:,~idx))=Rf(kk_time_idx(:,~idx))...
                        +dict(:,kk)*atoms(atom_idx(~idx),3)';
                    atoms(atom_idx(~idx),3)=0;
                end
                if Do_nonneg
                    %Rank-1 semi-nonnegative update
                    new_amps=XX'*dict(:,kk);
                    v=eps+new_amps.*(new_amps>0);
                    uk=XX*v;uk=uk/norm(uk);
                    ukk=uk*0;
                    while abs(uk'*ukk)<1-1e-6
                        ukk=uk;
                        v=max(eps,XX'*uk);
                        uk=XX*v;uk=uk/norm(uk);
                    end
                    dict(:,kk)=uk;
                else
                    [dict(:,kk),~,~]=svds(XX,1);
                end
                %update Residual and Approximation
                Rf(kk_time_idx)=Rf(kk_time_idx)+(Dold(:,kk)-dict(:,kk))*atoms(kk_idx,3)';
                approx(kk_time_idx)=approx(kk_time_idx)+(dict(:,kk)-Dold(:,kk))*atoms(kk_idx,3)';
            end
        end
    end
    %the waveforms are converged
    %look for redundant or unused waveforms
    if Do_nonneg
        Dist=fasterXcorr(dict,'');
    else
        Dist=fasterXcorr(dict,'abs');
    end
    Dist(1:1+K:end)=inf;
    idx=find(Dist<redundant_threshold,1);
    if ~isempty(idx)
        while ~isempty(idx)%replace redundant waveforms
            [row,~]=ind2sub([K,K],idx);
            w=randn(N,1);%replace with new initialization
            dict(:,row)=w/norm(w);
            Dist(row,:)=inf;
            idx=find(Dist<redundant_threshold,1);
        end
        %fprintf(1,'outer %i: restarted a redundant waveform\n',outer_count);
        outer_count=outer_count+1;
        %rerun without redundant waveform replaced by new initialization
        %this can create an infinite loop!
    else
        atoms=atoms(abs(atoms(:,3))>0,:);% non-zero atoms
        if ~isempty(atoms)
            %resort by mean amplitude (aesthetics only)
            v=zeros(K,1);
            for k=1:K
                amps=atoms(atoms(:,2)==k,3);
                if ~isempty(amps)
                    v(k)=mean(abs(amps));
                end
            end
            [~,resrt]=sort(v,'descend');
            dict=dict(:,resrt);
            
            %Estimate rates
            wave_counts=histc(atoms(:,2),1:K);
            wave_rates=wave_counts*freq/L;
            %fprintf(1,'outer %i: ',outer_count);
            %fprintf(1,'%i, ',wave_counts);
            %fprintf(1,'\n');
            if all(wave_rates>=rate_thresh)
                break;
            else
                if all(wave_rates<rate_thresh)
                    %none of the waveforms is at the minimum rate
                    %hope this never happens
                    %fprintf('All waveforms below minimum rate.\n');
                    dict=dict(:,wave_counts>0);
                    break;
                end
                if all(wave_rates(wave_counts>0)>=rate_thresh)
                    %all of the used waveforms are at the minimum rate
                    dict=dict(:,wave_counts>0);
                    break;
                end
                min_count=min(wave_counts(wave_counts>0));
                %remove only the waves with the minimum rate
                dict=dict(:,wave_counts>min_count);
                outer_count=outer_count+1;
            end
        else %No non-zero atoms, restart and try again
            dict=randn(size(dict));
            outer_count=outer_count+1;
        end
    end
end
end

