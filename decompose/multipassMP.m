function [ atoms, residual ,approx ] = multipassMP( X,Dict,mp_func,Approximation_passes)
% Wrapper function for performing matching pursuit (or block-based version) mulitple times
%--------------
% Inputs:
%--------------
% X                      [vector] signal
% Dict                   [matrix] shift-invariant dictionary
% mp_func                [function handle] handled for instantiation of 
%                           matching pursuit (2 input arguments: X and Dict,
%                           3 output arguments: atoms,residual,approx)
% Approximation_passes   [scalar] number of passes, since each pass will use a maximum # of atoms
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
residual=X;
approx=0*X;
atoms=[];
for kk=1:Approximation_passes
    [atoms1, residual ,approx1] = mp_func(residual,Dict);
    approx=approx1+approx;
    atoms=cat(1,atoms,atoms1);
end
end
