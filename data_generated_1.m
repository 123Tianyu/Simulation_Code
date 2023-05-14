function [V_kl_MRC,V_kl_MMSE] = data_generated_1(Hhat_local,Sample_num,N,K,M,p)
%If only one transmit power is provided, use the same for all the UEs
if length(p) == 1
   p = p*ones(K,1);
end

%If no specific Level 1 transmit powers are provided, use the same as for
%the other levels
if nargin<12
    p1 = p;
end


%Store identity matrices of different sizes
eyeK = eye(K);
eyeN = eye(N);



%Diagonal matrix with transmit powers and its square root
Dp = diag(p);

V_kl_MMSE = zeros(M,N,K,Sample_num);
V_kl_MRC  = zeros(M,N,K,Sample_num);

%% Go through all channel realizations
for n = 1:Sample_num 
    for l = 1:M
        %Extract channel realizations from all UEs to AP l
        H_il = reshape(Hhat_local(l,:,:,n),[N K]);

        V_MRC = H_il;
        V_kl_MRC(l,:,:,n) = V_MRC;

        %Compute MMSE combining
        V_MMSE = ((H_il*Dp*H_il')+eyeN)\(H_il*Dp); 
        V_kl_MMSE(l,:,:,n) = V_MMSE;
    end

end    
    V_kl_MRC = permute(V_kl_MRC,[1,3,2,4]);   % M×K×N×Sample_num
    V_kl_MMSE = permute(V_kl_MMSE,[1,3,2,4]); % M×K×N×Sample_num
end