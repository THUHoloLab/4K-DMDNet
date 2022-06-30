function [ax1,ax2,lineLossNpcc,lineLossTotal,lineLossTV,lineLossPercept]=initializePlots()   %定义用于训练过程显示的函数

set(0,'defaultfigurecolor','w')

% Initialize training progress plot.
fig1 = figure;

% Double the width and height of the figure.
% fig1.Position(3:4) = 2*fig1.Position(3:4);

ax1 = subplot(2,3,1:3);

% Plot the three losses on the same axes.
hold on
lineLossTotal = animatedline('Color',[0 0 0]);
lineLossNpcc = animatedline('Color','b');
lineLossPercept = animatedline('Color','r');
lineLossTV = animatedline('Color','g');

% Customize appearance of the graph.
legend('NPCC loss','Total Loss','TV loss','Perceptual Loss','Location','Southwest');
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

% Initialize image plot.
ax2 = subplot(2,3,4:6);
axis off

end