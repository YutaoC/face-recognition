function K = kernel(x1,x2,func_type,kernel_params)
%
% This function implement three different kernels which are
% Linear Kernel
% Radial Basis Function (RBF) Kernel
% Polynominal Kernel
%
% K = kernel(x1,x2,func_type,kernal_params)
%
% x1 - data set one
% x2 - data set two
% func_type - 'l' Linear Kernel
%             'g' Radial Basis Function (RBF) Kernel
%             'p' Polynominal Kernel
% kernel_params - kernel parameter
%
% Yutao Chen
% 16/11/2018
%
    [~,n1] = size(x1);
    [~,n2] = size(x2);
    K = zeros(n2,n1);
    if func_type == 'g'
        for j = 1:n1
            for i = 1:n2
                K(i,j) = exp(-norm(x1(:,j)-x2(:,i))/kernel_params); %RBF
            end
        end
    elseif func_type == 'p'
        for j = 1:n1
            for i = 1:n2
                K(i,j) = (x1(:,j)' * x2(:,i)+1).^kernel_params; %polynomial
            end
        end
    elseif func_type == 'l'
        for j = 1:n1
            for i = 1:n2
                K(i,j) = x1(:,j)' * x2(:,i); %linear
            end
        end
    end
end
