clear;
b_angle_moving = importdata('data/bird_angle_moving_112.5.txt');
b_angle_static = importdata('data/bird_angle_static_112.5.txt');
m_angle_moving = importdata('data/mavik_angle_moving_112.5.txt');
m_angle_static = importdata('data/mavik_angle_static_112.5.txt');
%%
set(gca,'fontname','times new roman'); 

t = tiledlayout(1,2,'TileSpacing','none');
t.TileSpacing = 'compact';

nexttile;
plot(b_angle_moving,'r');
hold on;
plot(b_angle_static, 'b');
grid on;
ax = gca;
ax.GridLineWidth = 2;
ax.YAxis.FontSize = 10;
ax.XAxis.FontSize = 10;
lg1 = legend('flapping state', 'static state','fontsize',13, 'fontname', ...
            'times new roman', 'location', 'northeast');
set(lg1,...
    'Position',[0.229166662544674 0.719777774598864 0.242333337084452 0.148333336512248],...
    'FontSize',13,...
    'FontName','times new roman');
%xlabel('Time [ms]','fontsize',16,'interpreter', 'latex');
ylabel('Phase [$^\circ$]','fontsize',16,'interpreter', 'latex');
title(' Bionic bird', 'FontSize',15,'FontName','times new roman');
ylim([65, 140]);
%exportgraphics(f,'figures/bird_static_moving.png', 'Resolution',800);

nexttile;


plot(m_angle_moving,'r');
hold on;
plot(m_angle_static, 'b');
grid on;
ax = gca;
ax.GridLineWidth = 2;
ax.YAxis.FontSize = 10;
ax.XAxis.FontSize = 10;
lg2 = legend('rotational state', 'static state','fontsize',13, ...
    'fontname', 'times new roman', 'location', 'northeast');
set(lg2,...
    'Position',[0.647555551454756 0.719777774598864 0.2570000041008 0.148333336512248],...
    'FontSize',13,...
    'FontName','times new roman');
xlabel(t, 'Time [ms]','fontsize',15,'interpreter', 'latex');
%ylabel('Phase[$^\circ$]','fontsize',16,'interpreter', 'latex');
ylim([-130, -55])
title('DJI mavic', 'FontSize',15,'FontName','times new roman');
%ax.YAxis.FontSize = 15;
%ax.XAxis.FontSize = 15;

set(gcf,'Position',[100 100 600 300]);
exportgraphics(gcf, 'figures/drone_bird_phase_time.png', 'Resolution',800);
