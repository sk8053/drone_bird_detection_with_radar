clear;
% http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/signal/specgram.html
b_angle_static = importdata('data/bird_angle_static_0.txt');
m_angle_static = importdata('data/mavik_angle_static_0.txt');

%angle_list = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180];
ang = 112.5;


file_name = compose("data/sampled_phase_data/mavik_angle_moving_%s.txt", [string(ang)]);
data_mavik =  importdata(file_name);
file_name = compose("data/sampled_phase_data/bird_angle_moving_%s.txt", [string(ang)]);
data_bird =  importdata(file_name);


R =256; % block length
% A narrow-band spectrogram is one computed using a relatively long block length R, (long window function).
% A wide-band spectrogram is one computed using a relatively short block length R, (short window function).
window = hamming(R);
 
N_fft = 512;% FFT length, This value determines the frequencies at which the discrete-time Fourier transform is computed
L = 32; % tme lapse between blocks, This is sometimes called the hop size
fs = 8000; % sampling frequency
overlap = R- L; % the number of samples by which the sections overlap

%If n is even, specgram returns nfft/2+1 rows (including the zero and Nyquist frequency terms). 
% If n is odd, specgram returns nfft/2 rows 

% The number of columns in B is 
% k = fix((n-numoverlap)/(length(window)-numoverlap))

% returns frequency and time vectors f and t respectively. t is a column vector of scaled times, with length equal to the number of columns of B.
% t(j) is the earliest time at which the jth window intersects a. t(1) is always equal to 0.
% so t(j) = L*j/fs
% length of t = (data length - R)/L +1
% t = [0, 1*L/fs, 2*L/fs, 3*L/fs, ..., k*L/fs]  k = (data length - R) / L +1
 
[B,f,t] = specgram(data_bird,N_fft,fs,window,overlap); 

tile = tiledlayout(1,2,'TileSpacing','none');
tile.TileSpacing = 'compact';

nexttile;
set(gca,'fontname','times new roman'); 
imagesc(t*fs,2*pi*f/fs,10*log10(abs(B)));
colormap('jet')
%colorbar();
axis xy
ax = gca;
ax.GridLineWidth = 2;
ax.YAxis.FontSize = 10;
ax.XAxis.FontSize = 10;
%xlabel ('Time (ms)', 'fontsize', 16, 'fontname', 'times new roman')
ylabel ({'Normalized frequency'; '(rad/sample)'}, 'fontsize', 16, ...
           'fontname', 'times new roman', 'Interpreter','latex');
title(' Bionic bird', 'FontSize',16,'FontName','times new roman');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[B,f,t] = specgram(data_mavik,N_fft,fs,window,overlap); 
nexttile;

imagesc(t*fs,2*pi*f/fs,10*log10(abs(B)));
colormap('jet')
c =colorbar();
c.FontSize = 10;
axis xy
ax = gca;
ax.GridLineWidth = 2;
ax.YAxis.FontSize = 10;
ax.XAxis.FontSize = 10;
xlabel (tile, 'Time [ms]', 'fontsize', 16, 'fontname', 'times new roman');
title('DJI mavic', 'FontSize',16,'FontName','times new roman');

set(gcf,'Position',[100 100 600 300]);
%yticklabels([]);
%ylabel ('Normalized frequencty (rad/sample)', 'fontsize', 16, ...
%           'fontname', 'times new roman', 'Interpreter','latex');
exportgraphics(gcf, 'figures/spectrogram_drone_and_bird.png', 'Resolution',800);
 