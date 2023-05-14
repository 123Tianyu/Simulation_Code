function Ru_cf_k = Rate_caculate_M(Hhat_mk,V_mk,K,M,yinta,pu_cf) 
%% 该函数仅适合计算接入点多天线时的速率
Hhat_mk = squeeze(Hhat_mk);     % M×N×K
V_mk    = squeeze(V_mk);        % M×N×K
Ru_cf_k = zeros(1,K);
%% 计算速率
    for k = 1:K
%         Ru_cf_k_fenzi = pu_cf*yinta(k)*abs(sum(V_mk(:,k).*Hhat_mk(:,k)))^2;
        temp1 = 0;
        for m = 1:M
            temp1 = temp1 + V_mk(m,:,k)'*Hhat_mk(m,:,k);
        end

        Ru_cf_k_fenzi = pu_cf*yinta(k)*norm(temp1)^2;


        Ru_cf_k_fenmu_left = 0;
        temp2 = 0;
        for m = 1:M
%             temp2 = temp2 + norm(V_mk(m,k))^2;
            temp2 = temp2 + V_mk(m,:,k)';
        end
        
%         Ru_cf_k_fenmu_right = abs(temp2)^2;
        Ru_cf_k_fenmu_right = norm(temp2)^2;

        for k1 = 1:K
           if(k1~=k)
               temp3 = 0;
               for m = 1:M
                   temp3 = temp3 + V_mk(m,:,k)'*Hhat_mk(m,:,k1);
               end
%                Ru_cf_k_fenmu_left = Ru_cf_k_fenmu_left + yinta(k1)*abs(temp1)^2;
               Ru_cf_k_fenmu_left = Ru_cf_k_fenmu_left + yinta(k1)*norm(temp3)^2;
           end
        end

        Ru_cf_k_fenmu_left = pu_cf*(Ru_cf_k_fenmu_left);

        Ru_cf_k_fenmu = Ru_cf_k_fenmu_left + Ru_cf_k_fenmu_right;
        Ru_cf_k(k) = log2(1 + Ru_cf_k_fenzi/Ru_cf_k_fenmu); 
    end
end