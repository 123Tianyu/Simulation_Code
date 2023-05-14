function [V_MRC_global,V_MMSE_global] = data_generated_2(Hhat_local,nbrOfRealizations,N,K,M,p)
%If only one transmit power is provided, use the same for all the UEs
if length(p) == 1
   p = p*ones(K,1);
end

%Store identity matrices of different sizes
eyeNL = eye(N*M);

%Diagonal matrix with transmit powers and its square root
Dp = diag(p);

Hhat = permute(Hhat_local,[1,3,4,2]);
Hhat1 = zeros(M*N,nbrOfRealizations,K);
for l = 1:M
   Hhat1((l-1)*N+1:l*N,:,:) = Hhat(l,:,:,:); 
end
V_MMSE_global = zeros(M*N,nbrOfRealizations,K);
V_MRC_global  = zeros(M*N,nbrOfRealizations,K);
%% Go through all channel realizations
for n = 1:nbrOfRealizations
    %Level 4
    %Extract channel estimate realizations from all UEs to all APs
    Hhatallj = reshape(Hhat1(:,n,:),[N*M K]);
    
    %Compute MR combining
    V_MRC = Hhatallj;
    V_MRC_global(:,n,:) = V_MRC;
    %Compute MMSE combining
    V_MMSE = ((Hhatallj*Dp*Hhatallj')+eyeNL)\(Hhatallj*Dp);
    V_MMSE_global(:,n,:) = V_MMSE;
end
