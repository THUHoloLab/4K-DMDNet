clear;close all;clc
addpath('./functions');
%% load network
load Untrained_4KDMDNet_Fresnel_30cm_520nm.mat
%% dataset
% load dataset
rawImagePath = './DIV2K_train_HR';
imds = imageDatastore(rawImagePath,'IncludeSubfolders',true);
augimds = augmentedImageDatastore([1080 1920],imds,'ColorPreprocessing',"rgb2gray");
% initialize plot
[ax1,ax2,lineLossNpcc]=initializePlots();
plotFrequency = 5;
%% training parameters
numEpochs = 10;  
miniBatchSize = 2;  
augimds.MiniBatchSize = miniBatchSize;
averageGrad = [];
averageSqGrad = [];
numIterations = floor(augimds.NumObservations*numEpochs/miniBatchSize)*10;
mbq = minibatchqueue(augimds,'MiniBatchSize',miniBatchSize,'MiniBatchFormat','SSBC');
learnRate = 0.001;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
executionEnvironment = "auto";
%% training
iteration = 0;
pic_num = 1;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)
        
        % Read mini-batch of data.
        [dlX] = next(mbq); 
  
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end

        for k=1:10
         iteration = iteration + 1;
         
        % Evaluate model gradients. 
         [gradients,dlYm,loss,lossNpcc] = dlfeval(@modelGradients,dlnet,dlX);

        % Update the network parameters using the Adam optimizer.
        [dlnet,averageGrad,averageSqGrad] = ...
            adamupdate(dlnet,gradients,averageGrad,averageSqGrad,iteration,...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
       
        addpoints(lineLossNpcc,iteration,double(gather(extractdata(lossNpcc))))
  
        % Every plotFequency iterations, plot the training progress.
        if mod(iteration,plotFrequency) == 0            
            % Use the first image of the mini-batch as a validation image.
            dlV = dlX(:,:,:,1);
            % Use the transformed validation image computed previously.
            dlVY = dlYm(:,:,:,1);
            dlVY = rescale(dlVY,0,255);
            dlZ = forward(dlnet,dlX,'Outputs','tanh');
            dlVZ = dlZ(:,:,:,1);
            dlVZ = rescale(dlVZ,0,255);
            
            % To use the function imshow, convert to uint8.
            validationImage = uint8(gather(extractdata(dlV)));
            transformedValidationImage = uint8(gather(extractdata(dlVY)));
            phaseImage = uint8(gather(extractdata(dlVZ)));
            
            % Plot the input image and the output image and increase size
            imshow(imtile({validationImage,transformedValidationImage,phaseImage},'GridSize', [1 3]),'Parent',ax2);
        end
        
        % Display time elapsed since start of training and training completion percentage.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        completionPercentage = round(iteration/numIterations*100,2);
        title(ax1,"Epoch: " + epoch + ", Iteration: " + iteration +" of "+ numIterations + "(" + completionPercentage + "%)"+...
            ", LearnRate: "+ learnRate + ", Elapsed: " + string(D))
        drawnow
        
       end
      
    end
    
    learnRate=learnRate*0.9;

end

save('Trained_4KDMDNet_Fresnel_30cm_520nm.mat','dlnet','averageGrad','averageSqGrad');
%% loss function
function [gradients,dlYm,loss,lossNpcc] = modelGradients(dlnet,dlX)

    [dlY] = forward(dlnet,dlX,'Outputs','I');
    dlYm = dlY(3097:4584,2519:5162,:,:);   
    X = gather(extractdata(dlX));
    Xc = imresize(X,[1488 2644]); 
    dlXc = dlarray(Xc, 'SSCB');
    lossNpcc = npccLoss(dlYm,dlXc);
    lossNpcc = (lossNpcc + 1)/2;
    
    % Calculate the total loss.
    loss = lossNpcc;

    gradients = dlgradient(loss,dlnet.Learnables);
end
function loss = npccLoss(dlX,dlY)

X0 = dlX - mean(dlX,[1 2]);
Y0 = dlY - mean(dlY,[1 2]);
X0_norm = sqrt(sum(X0.^2,[1 2]));
Y0_norm = sqrt(sum(Y0.^2,[1 2]));

npcc = -sum(X0.*Y0,[1 2])./(X0_norm.*Y0_norm);
loss = mean(npcc,'all');
end