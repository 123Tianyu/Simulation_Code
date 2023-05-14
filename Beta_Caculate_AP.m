function [Beta,AP_site,User_site] = Beta_Caculate_AP(M,K,D,L,d0,d1,deta_sh)
% 该函数用于产生AP位于一个边长为D的正方形网格上，AP位置固定，用户随机分布的大尺度衰落
% 这个正方形区域的中心就是原点，因此各个象限边界到原点距离均为 D/2
% 为了使得AP能够均匀分布在正方形的网格上，因此AP数量应该可以被开方

        AP_num_onGrid = sqrt(M); % 正方形边上的AP数
        Distance = zeros(M,K);
        PL_mk = zeros(M,K);
        Z_mk = rand(M,K);
        Beta = zeros(M,K);
%         AP_site = repmat( repmat( -D/2 : (D/(AP_num_onGrid-1)) : D/2 , [AP_num_onGrid 1] ) , [1 2]);  % 将正方形区域划分成等距的网格，对应小正方形的顶点坐标作为AP坐标
        AP_site_x = repmat( -D/2 : (D/(AP_num_onGrid-1)) : D/2 , [AP_num_onGrid 1] );  % 将正方形区域划分成等距的网格，对应小正方形的顶点坐标作为AP坐标  AP_num_onGrid×1
        AP_site_y = AP_site_x.';
        AP_site = zeros(M,2);
        for m = 1:M
            AP_site(m,1) = AP_site_x(m);
            AP_site(m,2) = AP_site_y(m);
        end

        User_site = D*rand(K,2)-D/2; % 生成[-D/2,D/2]的随机变量矩阵作为用户坐标
        for m = 1:M
            for k = 1:K
                Distance(m,k) = sqrt((AP_site(m,1)-User_site(k,1))^2 + (AP_site(m,2)-User_site(k,2))^2);
                if(Distance(m,k)>d1)
                   PL_mk(m,k) = -L - 35*log10(Distance(m,k));
                else
                    if(d0 <= Distance(m,k) <= d1)
                        PL_mk(m,k) = -L - 15*log10(d1) - 20*log10(Distance(m,k));
                    else
                        if(Distance(m,k) <= d0)
                            PL_mk(m,k) = -L - 15*log10(d1) - 20*log10(d0);
                        end
                    end
                end
                PL_mk(m,k) = 10^(PL_mk(m,k)/10);
                Beta(m,k) = PL_mk(m,k)*10^((deta_sh*Z_mk(m,k))/10);
                                
            end
        end
end