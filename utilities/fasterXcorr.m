function D=fasterXcorr(X,method)
[n,m]=size(X);
X=bsxfun(@rdivide,X,sqrt(sum(X.^2)));
X2=cat(1,zeros(n-1,m),X,zeros(n-1,m));
fX2=fft(X2);
if 1~=strcmp(method,'hilbert')
    ufX2=fft(flipud(X2));
else
    ufX2=fft(flipud(hilbert(X2)));
end
%C=sqrt(abs(ufX2/sqrt(n2))'*abs(fX2/sqrt(n2)));
% |IFFT(x.*y)|_inf=|x.*y|_inf<=|x.*y|_2=sqrt{ \sum_i abs(x_i y_i)^2<=sqrt{\sum_i abs(x_i) abs(y_i) <=|x^2|_2 |y^2|_2  

block_size=50;
pdistout=zeros(m*(m-1)/2,1);
ii=0;
for col=1:m-1
    for rowstart=col+1:block_size:m
    rows=rowstart:min(rowstart+block_size-1,m);
    H=bsxfun(@times,ufX2(:,col),fX2(:,rows));
    val=ifft(H);
switch method
case {'abs','hilbert'}
    val=abs(val);
otherwise
end
    val=max(val,[],1);
%    val=max(filter(flipud(X(:,col)),1,X2(:,rows)));
    pdistout(ii+(1:numel(rows)))=1-val;
    ii=ii+numel(rows);
    end
end
D=squareform(pdistout);
D(1:m+1:end)=1-sum(X.^2);
end