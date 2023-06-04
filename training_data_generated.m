clc
close all % 关闭所有的Figure窗口 
clearvars % 清除工作区间的变量
tic

%% 样本数设置
Sample_num = 2*1e3; 

%% 模型参数
M = 9;      % AP个数
N = 1;      % AP天线数
K = 4;      % 用户数

%% 三段式大尺度衰落参数设置
D = 1;      % 正方形区域边长（单位：千米）
d0 = 10e-3; % （单位：千米）
d1 = 50e-3; % （单位：千米）

f = 1.9e3;       % 载频：MHz
h_AP = 15;       % AP天线高度（单位：米）
h_u = 1.65;      % 用户天线高度（单位：米）
L = 46.3 + 33.9*log10(f) - 13.82*log10(h_AP) - (1.1*log10(f)-0.7)*h_u + (1.56*log10(f)-0.8);

deta_sh_dB = 8;
deta_sh = 10^(deta_sh_dB/10); % 将db形式变为十进制数值

tau_c = 200; % 相干间隔长度
tau_p = K; % 导频长

yinta = ones(1,K);
k_B = 1.381e-23;
T0 = 290;
noise_figure_dB = 9; % dB
noise_figure = 10^(noise_figure_dB/10);
B = 20e6; % MHz
noise_power = B*k_B*T0*noise_figure;
Pu_cf = 100e-3; % 单位 W
pu_cf = Pu_cf/noise_power;

Ru_cf_k_MRC_local = zeros(Sample_num,K);
Ru_cf_k_MMSE_local = zeros(Sample_num,K);

Ru_cf_k_MRC_global = zeros(Sample_num,K);
Ru_cf_k_MMSE_global = zeros(Sample_num,K);

%Generate one setup with UEs at random locations
H_local = zeros(M,K,N,Sample_num);
[Beta,AP_Site,User_Site] = Beta_Caculate_AP(M,K,D,L,d0,d1,deta_sh);  % 获取大尺度
H = sqrt(1/2)*(randn(M,K,N,Sample_num)+1j*randn(M,K,N,Sample_num));
for m = 1:M
    for k = 1:K
        H_local(m,k,:,:) = H_local(m,k,:,:) + sqrt(Beta(m,k))*H(m,k,:,:);
    end
end


[V_MRC_local,V_MMSE_local] = data_generated_1(H_local,Sample_num,N,K,M,pu_cf); % M×K×N×Sample_num
[V_MRC_global,V_MMSE_global] = data_generated_2(H_local,Sample_num,N,K,M,pu_cf); % V_MMSE_global MN×Sample_num×K
H1= permute(H_local,[4,1,3,2]);

H_mk = permute(H_local,[1,3,2,4]);
V_MRC_Local = permute(V_MRC_local,[1,3,2,4]);   % M×N×K×Sample_num
V_MMSE_Local = permute(V_MMSE_local,[1,3,2,4]); % M×N×K×Sample_num

temp4 = zeros(M,N,Sample_num,K);  % 这步后面也有用的
for m = 1:M
    temp4(m,:,:,:) = V_MMSE_global((m-1)*N+1:m*N,:,:);
end
V_MMSE_Global = permute(temp4,[1,2,4,3]); % M×N×K×Sample_num

for n = 1:Sample_num
    %Display simulation progress
    disp(['序号 ' num2str(n) ' out of ' num2str(Sample_num)]);
    if (N>1)  % 多天线速率计算
        Ru_cf_k_MRC_local(n,:) = Rate_caculate_M(H_mk(:,:,:,n),V_MRC_Local(:,:,:,n),K,M,yinta,pu_cf);
        Ru_cf_k_MMSE_local(n,:) = Rate_caculate_M(H_mk(:,:,:,n),V_MMSE_Local(:,:,:,n),K,M,yinta,pu_cf);
        Ru_cf_k_MMSE_global(n,:) = Rate_caculate_M(H_mk(:,:,:,n),V_MMSE_Global(:,:,:,n),K,M,yinta,pu_cf);
    else      % 单天线速率计算
        Ru_cf_k_MRC_local(n,:) = Rate_caculate_S(H_mk(:,:,:,n),V_MRC_Local(:,:,:,n),K,M,yinta,pu_cf);
        Ru_cf_k_MMSE_local(n,:) = Rate_caculate_S(H_mk(:,:,:,n),V_MMSE_Local(:,:,:,n),K,M,yinta,pu_cf);
        Ru_cf_k_MMSE_global(n,:) = Rate_caculate_S(H_mk(:,:,:,n),V_MMSE_Global(:,:,:,n),K,M,yinta,pu_cf);
    end
end

%% 计算和速率
Ru_cf_k_MRC_SUM = sum(Ru_cf_k_MRC_local,2);
Ru_cf_k_MMSE_SUM = sum(Ru_cf_k_MMSE_local,2);
Ru_cf_k_MMSE_SUM1 = sum(Ru_cf_k_MMSE_global,2);

%% 数据处理 按照AP划分数据
Hhat_local_real_MMSE = real(H_local);
Hhat_local_imag_MMSE = imag(H_local);
Hhat_local_abs_MMSE  = abs(H_local);
Hhat_data_CNN_MMSE   = zeros(M,K,N,Sample_num,3);
Hhat_data_CNN_MMSE(:,:,:,:,1) = Hhat_local_real_MMSE;
Hhat_data_CNN_MMSE(:,:,:,:,2) = Hhat_local_imag_MMSE;
Hhat_data_CNN_MMSE(:,:,:,:,3) = Hhat_local_abs_MMSE;
Hhat_data_CNN_MMSE = permute(Hhat_data_CNN_MMSE,[1,4,2,3,5]); % M×Sample_num×K×N×3

Hhat_data_DNN_MMSE = Hhat_data_CNN_MMSE;

V_MMSE_Local = permute(V_MMSE_local,[1,4,2,3]); % M×Sample_num×K×N

temp2 = zeros(M,Sample_num,K*N);
for k = 1:K
   temp2(:,:,(k-1)*N+1 : k*N) = V_MMSE_Local(:,:,k,:);
end
V_MMSE_local_real = real(temp2);
V_MMSE_local_imag = imag(temp2);
Vkl_data_CNN_MMSE = zeros(M,Sample_num,2*K*N);
Vkl_data_CNN_MMSE(:,:,1 : K*N) = V_MMSE_local_real;
Vkl_data_CNN_MMSE(:,:,K*N+1 : 2*K*N) = V_MMSE_local_imag;

V_MRC_Local = permute(V_MRC_local,[1,4,2,3]); % M×Sample_num×K×N
temp3 = zeros(M,Sample_num,K*N);
for k = 1:K
   temp3(:,:,(k-1)*N+1 : k*N) = V_MRC_Local(:,:,k,:);
end
V_MRC_local_real = real(temp3);
V_MRC_local_imag = imag(temp3);
Vkl_data_CNN_MRC = zeros(M,Sample_num,2*K*N);
Vkl_data_CNN_MRC(:,:,1 : K*N) = V_MRC_local_real;
Vkl_data_CNN_MRC(:,:,K*N+1 : 2*K*N) = V_MRC_local_imag;

Hhat_local_DNN_MMSE = permute(H_local,[1,4,2,3]);
temp1 = zeros(M,Sample_num,K*N);
for k = 1:K
   temp1(:,:,(k-1)*N+1 : k*N) = Hhat_local_DNN_MMSE(:,:,k,:);
end
Hhat_local_DNN_real_MMSE = real(temp1);
Hhat_local_DNN_imag_MMSE = imag(temp1);
Hhat_local_DNN_abs_MMSE  = abs(temp1);
Hhat_data_DNN_MMSE = zeros(M,Sample_num,3*K*N);
Hhat_data_DNN_MMSE(:,:,1 : K*N) = Hhat_local_DNN_real_MMSE;
Hhat_data_DNN_MMSE(:,:,K*N+1 : 2*K*N) = Hhat_local_DNN_imag_MMSE;
Hhat_data_DNN_MMSE(:,:,2*K*N+1 : 3*K*N) = Hhat_local_DNN_abs_MMSE;

Hhat_data_DNN_MRC = Hhat_data_DNN_MMSE;

Vkl_data_DNN_MMSE = Vkl_data_CNN_MMSE;
Vkl_data_DNN_MRC  = Vkl_data_CNN_MRC;


temp4 = reshape(permute(temp4,[1,3,4,2]),[M,Sample_num,K,N]); % M,Sample_num,K,N,
temp5 = zeros(M,Sample_num,K*N);
V_MMSE_global_DNN = zeros(M,Sample_num,2*K*N);
for k = 1:K
   temp5(:,:,(k-1)*N+1:k*N) = temp4(:,:,k,:); 
end
V_MMSE_global_DNN_real = real(temp5);
V_MMSE_global_DNN_imag = imag(temp5);
V_MMSE_global_DNN(:,:,1 : K*N) = V_MMSE_global_DNN_real;
V_MMSE_global_DNN(:,:,K*N+1 : 2*K*N) = V_MMSE_global_DNN_imag;


%% 画图
figure
hold on;
plot(AP_Site(:,1),AP_Site(:,2),'rp');
plot(User_Site(:,1),User_Site(:,2),'bo');
axis square;
title('五角星表示AP，圆圈表示用户');
grid on
grid minor
box on;

figure;
hold on; box on;
plot(sort(reshape(Ru_cf_k_MRC_SUM,[Sample_num 1])),linspace(0,1,Sample_num),'b-','LineWidth',2);
plot(sort(reshape(Ru_cf_k_MMSE_SUM,[Sample_num 1])),linspace(0,1,Sample_num),'r-','LineWidth',2);
plot(sort(reshape(Ru_cf_k_MMSE_SUM1,[Sample_num 1])),linspace(0,1,Sample_num),'g-','LineWidth',2);
xlabel('Spectral efficiency [bit/s/Hz]','Interpreter','Latex');
ylabel('CDF','Interpreter','Latex');
legend('MRC','MMSE-local','MMSE-global','Location','south');
toc

