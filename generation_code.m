clc;clear;close all
addpath('./functions');
%% Encoder network load
load UNet.mat
lgraph=layerGraph(dlnet);
%% Decoder generation:Fresnel
lamda = 520e-9;   %450nm 520nm 638nm
k = 2*pi/lamda;  
z = 0.3;
pix = 3.74e-6;
holoSize = [2160 3840];
outputSize = [7680 7680];
Lx = pix*holoSize(2);Ly = pix*holoSize(1);
[x,y] = meshgrid(-Lx/2:pix:Lx/2-pix,-Ly/2:pix:Ly/2-pix);
P = pi*(x.^2 + y.^2)/(lamda*z);   %spherical phase

% spherical phase addition
lgraph = addLayers(lgraph,plusLayer(P,'plus'));
lgraph = connectLayers(lgraph,'tanh','plus');

% split
lgraph = addLayers(lgraph,cosLayer('cos'));
lgraph = addLayers(lgraph,sinLayer('sin'));
lgraph = connectLayers(lgraph,'plus','cos');
lgraph = connectLayers(lgraph,'plus','sin');

% zero padding
lgraph = addLayers(lgraph,ZeroPadding2dLayer('cospad', (outputSize - holoSize)/2));
lgraph = addLayers(lgraph,ZeroPadding2dLayer('sinpad', (outputSize - holoSize)/2));
lgraph = connectLayers(lgraph,'cos','cospad');
lgraph = connectLayers(lgraph,'sin','sinpad');

% fftshift
lgraph = addLayers(lgraph,fftshiftLayer('cosshift'));
lgraph = addLayers(lgraph,fftshiftLayer('sinshift'));
lgraph = connectLayers(lgraph,'cospad','cosshift');
lgraph = connectLayers(lgraph,'sinpad','sinshift');

% fft2
lgraph = addLayers(lgraph,fft2DLayer('Fcos'));
lgraph = addLayers(lgraph,fft2DLayer('Fsin'));
lgraph = connectLayers(lgraph,'cosshift','Fcos');
lgraph = connectLayers(lgraph,'sinshift','Fsin');
lgraph = addLayers(lgraph,subtractionLayer('Fr'));
lgraph = addLayers(lgraph,additionLayer(2,'Name','Fi'));
lgraph = connectLayers(lgraph,'Fcos/real','Fr/in1');
lgraph = connectLayers(lgraph,'Fsin/imag','Fr/in2');
lgraph = connectLayers(lgraph,'Fcos/imag','Fi/in1');
lgraph = connectLayers(lgraph,'Fsin/real','Fi/in2');

% fftshift
lgraph = addLayers(lgraph,fftshiftLayer('Frf'));
lgraph = addLayers(lgraph,fftshiftLayer('Fif'));
lgraph = connectLayers(lgraph,'Fr','Frf');
lgraph = connectLayers(lgraph,'Fi','Fif');     
            
% intensity
lgraph = addLayers(lgraph,intensityLayer('I'));
lgraph = connectLayers(lgraph,'Frf','I/in1');
lgraph = connectLayers(lgraph,'Fif','I/in2');

dlnet = dlnetwork(lgraph);
save('Untrained_4KDMDNet_Fresnel_30cm_520nm.mat','dlnet'); 