h00_conv = importdata('data/h00_conv.txt');
h01_conv = importdata('data/h01_conv.txt');
h10_conv = importdata('data/h10_conv.txt');
h11_conv = importdata('data/h11_conv.txt');
SNRs = importdata('data/SNRs.txt');

figure()

t = tiledlayout(1,2,'TileSpacing','none');
t.TileSpacing = 'compact';

nexttile;

set(gca,'fontname','times new roman');  
plot(SNRs, h00_conv, "r--^", 'LineWidth',1.2);
hold on;
plot(SNRs, h10_conv, "gp-", 'LineWidth',1.2);
hold on;

plot(SNRs, h01_conv, 'bs-.', 'LineWidth',1.2);
hold on;
plot(SNRs, h11_conv, 'ko:', 'LineWidth',1.2);

grid on;
%legend('P(0|0)', 'P(1|0)','P(0|1)','P(1|1)', ...
%    'location', 'southeast', 'fontsize', 14, ...
%    'fontname', 'times new roman');
yticks(linspace(0,1,11));
xticks(linspace(-5, 30, 8));
set(gca,'XTickLabelRotation',0);
ax = gca;
ax.GridLineWidth = 2;
ax.YAxis.FontSize = 12;
ax.XAxis.FontSize = 12;
%xlabel('SNR [dB]', 'fontname', 'times new roman', 'fontsize', 15);
ylabel('Probability', 'fontname', 'times new roman', 'fontsize', 15);
%exportgraphics(gcf,'figures/prob_vs_snr_conv.png', 'Resolution',800);
ti = title('Conv. classifier', 'fontname', 'times new roman', 'fontsize', 15);
%set(ti,'position',get(ti,'position')+[0 0.013 0]);
ylim([0,1.03]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h00_conv_lstm = importdata('data/h00_conv_lstm.txt');
h01_conv_lstm = importdata('data/h01_conv_lstm.txt');
h10_conv_lstm = importdata('data/h10_conv_lstm.txt');
h11_conv_lstm = importdata('data/h11_conv_lstm.txt');
SNRs = importdata('data/SNRs.txt');



nexttile;
plot(SNRs, h00_conv_lstm, "r--^", 'LineWidth',1.2);
hold on;
plot(SNRs, h10_conv_lstm, "gp-", 'LineWidth',1.2);
hold on;

plot(SNRs, h01_conv_lstm, 'bs-.', 'LineWidth',1.2);
hold on;
plot(SNRs, h11_conv_lstm, 'ko:', 'LineWidth',1.2);
yticks(linspace(0,1,11));
yticklabels([]);
xticks(linspace(-5, 30, 8));
set(gca,'XTickLabelRotation',0);
ylim([0,1.03]);
grid on;

legend('P(0|0)', 'P(1|0)','P(0|1)','P(1|1)', ...
    'location', 'southeast', 'fontsize', 14, ...
    'fontname', 'times new roman');
ax = gca;
ax.GridLineWidth = 2;
ax.YAxis.FontSize = 12;
ax.XAxis.FontSize = 12;
xlabel(t, 'SNR [dB]', 'fontname', 'times new roman', 'fontsize', 15);
ti = title('ConvLSTM classifier', 'fontname', 'times new roman', 'fontsize', 15);
%set(ti,'position',get(ti,'position')+[0 0.013 0]);
%ylabel('Probability', 'fontname', 'times new roman', 'fontsize', 15);
set(gcf,'Position',[100 100 600 300]);
exportgraphics(gcf,'figures/prob_vs_snr.png', 'Resolution',800);
