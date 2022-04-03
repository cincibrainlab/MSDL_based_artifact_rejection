function [atoms, residual ,approx] = blockbasedMP(X,Dict,max_occurences,block_starts,varargin)
% Performs matching pursuit
%--------------
% Inputs:
%--------------
% X                [vector] signal
% Dict             [matrix] shift-invariant dictionary
% max_occurences   [scalar] maximum # of atoms
% block_starts     [vector] the start indices of non-contiguous blocks
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
% residual    [vector] residual signal (internal to MP known as Rf)
% approx      [vector] approximation signal (internal to MP known as Ff)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set-up for default parameters

N=size(Dict,1);
Lo=length(X);
M=2*N-1;%block length
%L/M number of blocks
atoms=zeros(max_occurences,4);
kk=1;
X=cat(1,X(:),zeros(ceil(Lo/M)*M-Lo,1));
L=length(X);
residual=0*X;
approx=residual;
tts=[1,block_starts(:)',L];
for ii=1:numel(block_starts)+1
    tt=tts(ii):tts(ii+1);
    if numel(tt)<=N
        warning('Block length is shorter or equal to filter length\n');
    end
    [block_atoms, block_residual ,block_approx] = shiftInvariantMP(X(tt),Dict,round(numel(tt)/L*max_occurences),varargin);
    block_atoms(:,1)=block_atoms(:,1)-1+tt(1);
    atoms(kk+(0:size(block_atoms,1)-1),:)=block_atoms;
    kk=kk+size(block_atoms,1);
    residual(tt)=block_residual;
    approx(tt)=block_approx;
end

approx=approx(1:Lo);%trim to original length
residual=residual(1:Lo);%trim to original length
atoms=atoms(atoms(:,1)<=Lo & atoms(:,1)>0,:);
% sort atoms by magnitude and trim unused space for atoms
[val,idx]=sort(abs(atoms(:,3)),'descend');
idx=idx(val>0);
atoms=atoms(idx,:);
end

