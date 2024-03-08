acc_conv_lstm = importdata('data/plotting_data/train_accuraciy_conv_lstm.txt');
val_conv_lstm = importdata('data/plotting_data/val_accuraciy_conv_lstm.txt');

acc_conv = importdata('data/plotting_data/train_accuraciy_conv.txt');
val_conv = importdata('data/plotting_data/val_accuraciy_conv.txt');

%plot(acc_conv, 'r-*');
%hold on;
%plot(acc_conv_lstm, 'ro');
%hold on;

plot(val_conv, 'r-*','LineWidth', 1.5);
hold on;
plot(val_conv_lstm, 'k-o', 'LineWidth', 1.5);
grid on;
legend('CNN', 'Conv-LSTM', 'Location', 'southeast' , 'Fontsize', 14);
xlabel ('Epoch', 'Fontsize', 14);
ylabel ('Accuracy', 'Fontsize', 14);
set(gca,'fontsize',12);

f = gca;
exportgraphics(f,'acc_eval.png', 'Resolution',400);




