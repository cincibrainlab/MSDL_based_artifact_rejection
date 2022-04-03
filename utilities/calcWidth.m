function [ fcutlow,fcuthigh ] = calcWidth(X,ind)
ind=ind(:);
[maxX,maxID]=max(abs(X));
X=bsxfun(@rdivide,abs(X),maxX);


Xmax=bsxfun(@le,ind(maxID)',ind).*(X>.7071);
Xmin=bsxfun(@ge,ind(maxID)',ind).*(X>.7071);
fmax=repmat(ind,1,size(X,2));
fmin=repmat(ind,1,size(X,2));
fmax(~Xmax)=nan;
fmin(~Xmin)=nan;
fcutlow=min(fmin);
fcuthigh=max(fmax);

end

