clear;clc;close all;
addpath('./functions');
%% load trained network
load Trained_4KDMDNet_Fresnel_30cm_520nm.mat
%% load test image
X = imread('./DIV2K_valid_HR/0801.png'); 
X = imresize(X,[1080,1920]);
X = im2gray(X);
X = single(X);
dlX = gpuArray(dlarray(X,'SSCB')); 
%% predict hologram
tic
dlY = forward(dlnet,dlX,'Outputs','tanh');
toc
dlZ = forward(dlnet,dlX);
Y = gather(extractdata(dlY));
Z = gather(extractdata(dlZ));
figure,imshow(Y,[]);title('hologram')
figure,imshow(Z,[]);title('reconstruction')
imwrite(mat2gray(Y),'h(0801).bmp','bmp')