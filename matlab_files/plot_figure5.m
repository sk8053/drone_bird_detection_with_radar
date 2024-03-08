acc_conv = importdata('data/test_data_conv.txt');
SNRs = importdata('data/SNRs.txt');
acc_convlstm = importdata('data/test_data_conv_lstm.txt');

figure()
set(gca,'fontname','times new roman');  
plot(SNRs, acc_conv, "r--^", 'LineWidth',1.2);
hold on;
plot(SNRs, acc_convlstm, 'bo-.', 'LineWidth',1.2);
grid on;
legend('Conv. classifier', 'ConvLSTM classifier', ...
    'location', 'southeast', 'fontsize', 14, ...
    'fontname', 'times new roman');
ax = gca;
ax.GridLineWidth = 2;
ax.YAxis.FontSize = 12;
ax.XAxis.FontSize = 12;
xlabel('SNR [dB]', 'fontname', 'times new roman', 'fontsize', 15);
ylabel('Accuracy', 'fontname', 'times new roman', 'fontsize', 15);


exportgraphics(gcf,'figures/accuracy_vs_snr.png', 'Resolution',800);