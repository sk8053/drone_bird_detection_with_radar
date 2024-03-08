clear;
conv = importdata('robust_conv.txt');
conv_lstm = importdata('robust_conv_lstm.txt');

plot(conv(:,1), conv(:,2), 'ro-','LineWidth', 1.5);
hold on;
plot(conv_lstm(:,1), conv_lstm(:,2), 'k-^','LineWidth', 1.5);
grid on;
xlabel('Variance of noise ($\sigma^2$)', 'interpreter', 'latex', 'fontsize', 16, 'fontname','times new roman');
ylabel('Accuracy', 'fontsize', 16, 'fontname','times new roman')
ax.YAxis.FontSize = 15;
ax.XAxis.FontSize = 15;
legend ('CNN', 'Conv-LSTM', 'fontsize', 13, 'location', 'best', 'fontname','times new roman')
xticks(0:0.2:2)
ylim([0.5,1])
yticks(0.5:0.1:1)
%set(gca,'fontsize',12);
f = gca;
exportgraphics(f,'figures/comp_robustness.png', 'Resolution',800);
