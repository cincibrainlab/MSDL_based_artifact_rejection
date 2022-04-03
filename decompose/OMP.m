function [I,w, Rf ,Ff] = OMP(X,dict,Natoms)
% Orthogonal matching pursuit
%--------------
% Inputs: 
%--------------
% X         [vector] input signal
% dict      [matrix] initial vecctor dictionary
% Natoms    [scalar] how many vectors to include in the decomposition
%--------------
% Outputs: 
%--------------
% I      [vector] support (indices of selected vectors)
% w      [vector] coefficients of selected vectors 
% Rf     [vector] residual vector
% Ff     [vector] approximation vector 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rf=X;
A=zeros(size(X,1),Natoms);
w=zeros(Natoms,1);
I=zeros(Natoms,1);
for ii=1:Natoms
 CC=Rf'*dict;
 [~, idx]=max(abs(CC));
 A(:,ii)=dict(:,idx);
 I(ii)=idx;
 w(1:ii)=pinv(A(:,1:ii))*X;
 Ff=A(:,1:ii)*w(1:ii);
 Rf=X-Ff;
end


