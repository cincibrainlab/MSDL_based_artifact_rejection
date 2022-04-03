function [D,Lags]=fasterXcorr2(X,Y,method)
[n,m]=size(X);
[n2,m2]=size(Y);
X=bsxfun(@rdivide,X,sqrt(sum(X.^2)));
Y=bsxfun(@rdivide,Y,sqrt(sum(Y.^2)));

N=max(n,n2);
X=cat(1,X,zeros(N-n,m));
Y=cat(1,Y,zeros(N-n2,m2));
X2=cat(1,zeros(N-1,m),X,zeros(N-1,m));
Y2=cat(1,zeros(N-1,m2),Y,zeros(N-1,m2));

fX=fft(X2);
fY=fft(flipud(Y2));
%C=sqrt(abs(ufX2/sqrt(n2))'*abs(fX2/sqrt(n2)));
% |IFFT(x.*y)|_inf=|x.*y|_inf<=|x.*y|_2=sqrt{ \sum_i abs(x_i y_i)^2<=sqrt{\sum_i abs(x_i) abs(y_i) <=|x^2|_2 |y^2|_2


M=size(X2,1);
P=floor(M/2);
block_size=50;
D=zeros(m,m2);
Lags=zeros(m,m2);
for col=1:m
    for rowstart=1:block_size:m2
        rows=rowstart:min(rowstart+block_size-1,m2);
        H=bsxfun(@times,fX(:,col),fY(:,rows));
        %    val=ifftshift(ifft(H),1);
        val=ifft(H);
        switch method
            case 'abs'
                val=abs(val);
            otherwise
        end
        [val,lag]=max(val);
        lag=(lag<=P).*lag+(lag>P).*(lag-2*P-mod(M,2));
        D(col,rows)=1-val;
        Lags(col,rows)=lag;
    end
end
end