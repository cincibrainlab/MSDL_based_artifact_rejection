function [atoms, Rf ,Ff] = shiftInvariantMP(X,Dict,max_occurences,varargin)
% Performs matching pursuit
%--------------
% Inputs:
%--------------
% X                [vector] signal
% Dict             [matrix] shift-invariant dictionary
% max_occurences   [scalar] maximum # of atoms
% (optional arguments that MUST appear in this order)
% do_nonneg        [bool]  if true then non-negative constraint is imposed
% no_overlap       [bool] if true then no overlap of waveforms is imposed
% block_start_indices [vector] the start indices of non-contiguous blocks,
%                which(allows one to decompose concatenated trials, etc.)
%--------------
% Outputs:
%--------------
% atoms [matrix: m by 4 ]   number of rows = number of waveform occurences
%                           Atomic decomposition: each row is (i,j,a,e)
%                               i - time index
%                               j - waveform index
%                               a - coefficient
%                               e - squared error after adding occurence
% Rf    [vector] residual signal
% Ff    [vector] approximation signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set-up for default parameters
f=@(x) abs(x);
no_overlap=0;
block_start_indices=[];
nvarargin=numel(varargin);
if nvarargin==1 && iscell(varargin{1})
    varargin=varargin{1};
    nvarargin=numel(varargin);
end
for ii=1:nvarargin
    switch ii
        case 1
            do_nonneg=varargin{ii};
            if do_nonneg~=0
                f=@(x) x;
            end
            if numel(do_nonneg)==1 && ( islogical(do_nonneg) || ismember(do_nonneg,[ 0 1]))
            else
                ('Do non-negative should be a binary flag or logical value\n');
            end
        case 2
            no_overlap=varargin{ii};
            if numel(no_overlap)==1 && ( islogical(no_overlap) || ismember(no_overlap,[ 0 1]))
            else
                ('No overlap should be a binary flag or logical value\n');
            end
        case 3
            block_start_indices=varargin{ii};
    end
end
[N,K]=size(Dict);
Lo=length(X);
M=2*N-1;%block length
%L/M number of blocks
scalings=sqrt(sum(Dict.^2,1));% assumes they are all unit-norm
D=bsxfun(@rdivide,Dict,scalings);


NN2=round(N/2);
X=cat(1,X(:),zeros(ceil(Lo/M)*M-Lo,1));
L=length(X);
ns=L;
%[CC_self,wnorm] = compWprojW(D);
CC_self=reshape(xcorr(D),2*N-1,K,K);
Of=X(:);
Ff=0*Of;
sse=sum(Of.^2);%squared area
neg_c = conj(fftfilt(conj(D(:,:)),flipud(Of))); % negative lags
CC=flipud(neg_c(end-ns+1:end,:));
for c_idx=block_start_indices(:)'
    tt=max(1,c_idx-N+1):min(Lo,max(c_idx-1,1));
    CC(tt,:)=0;
end
count=sum(CC(:)>0);
atoms=zeros(max_occurences,4);
for ii=1:max_occurences
    [c,idx]=max(f(CC));
    [~,IDX1]=max(c); %max over bases
    npad=idx(IDX1)-1;
    LL=min(N,ns-npad);
    
    a=CC(npad+1,IDX1);
    part_sse=sse-sum((Of(npad+(1:LL))-Ff(npad+(1:LL))).^2);
    Ff(npad+(1:LL))=Ff(npad+(1:LL))+D(1:LL,IDX1)*a;
    new_sse=part_sse+sum((Of(npad+(1:LL))-Ff(npad+(1:LL))).^2);
    sse=new_sse;
    atoms(ii,:)=[npad+1,IDX1,a/scalings(IDX1),new_sse];
    
    %hard no overlap setting
    if no_overlap==0%soft_overlap
        nlpad=max(0,npad-N+1); %move it left by N-1
        LL2=LL+npad-nlpad;
        CC(nlpad+(1:LL2),:)=CC(nlpad+(1:LL2),:)-a*CC_self(max(1,N-npad):N-1+LL,:,IDX1);
    else
        nlpad=max(0,npad-NN2); %move it left by L/2
        LL2=min(N,ns-nlpad);
        count=count-sum(sum(CC(nlpad+(1:LL2),:)>0));
        CC(nlpad+(1:LL2),:)=0;% don't let any base appear in a L length window
        if count<=0% all(CC(:)==0)
            break;
        end
    end
end
Rf=Of-Ff;%amp*d;
nzatoms=abs(atoms(:,3))>0;
Ff=Ff(1:Lo);
Rf=Rf(1:Lo);
atoms=atoms(nzatoms & atoms(:,1)<=Lo,:);
end

