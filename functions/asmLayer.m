classdef asmLayer < nnet.layer.Layer %#codegen
   
    properties
        % imageConv block.
        Network
    end
    
    methods
        function layer = asmLayer(inputSize,Hr,Hi,NameValueArgs)
            
            % Parse input arguments.
            arguments
                inputSize
                Hr
                Hi
                NameValueArgs.Name = ''
            end
            name = NameValueArgs.Name;

            % Set number of inputs.
            layer.NumInputs = 1;
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Fourer intensity of phase-only input";
           
            % Define nested layer graph.
            lgraph = layerGraph;
            lgraph = addLayers(lgraph,imageInputLayer([inputSize 1],'Normalization','None','Name','in'));
                     
            lgraph = addLayers(lgraph,cosLayer('cos'));
            lgraph = addLayers(lgraph,sinLayer('sin'));
            lgraph = connectLayers(lgraph,'in','cos');
            lgraph = connectLayers(lgraph,'in','sin');
            
            % zero padding
            lgraph = addLayers(lgraph,ZeroPadding2dLayer('cospad', inputSize/2));
            lgraph = addLayers(lgraph,ZeroPadding2dLayer('sinpad', inputSize/2));
            lgraph = connectLayers(lgraph,'cos','cospad');
            lgraph = connectLayers(lgraph,'sin','sinpad');
            
            % fftshift
%             lgraph = addLayers(lgraph,fftshiftLayer('cosshift'));
%             lgraph = addLayers(lgraph,fftshiftLayer('sinshift'));
%             lgraph = connectLayers(lgraph,'cos','cosshift');
%             lgraph = connectLayers(lgraph,'sin','sinshift');
            
            % fft2
            lgraph = addLayers(lgraph,fft2DLayer('Fcos'));
            lgraph = addLayers(lgraph,fft2DLayer('Fsin'));
            lgraph = connectLayers(lgraph,'cospad','Fcos');
            lgraph = connectLayers(lgraph,'sinpad','Fsin');
            
            lgraph = addLayers(lgraph,subtractionLayer('Fr'));
            lgraph = addLayers(lgraph,additionLayer(2,'Name','Fi'));
            lgraph = connectLayers(lgraph,'Fcos/real','Fr/in1');
            lgraph = connectLayers(lgraph,'Fsin/imag','Fr/in2');
            lgraph = connectLayers(lgraph,'Fcos/imag','Fi/in1');
            lgraph = connectLayers(lgraph,'Fsin/real','Fi/in2');
            
            % Angular spectrum transfer function
            lgraph = addLayers(lgraph,hadamardProdLayer(Hr,'FrHr'));
            lgraph = addLayers(lgraph,hadamardProdLayer(Hr,'FiHr'));
            lgraph = addLayers(lgraph,hadamardProdLayer(Hi,'FrHi'));
            lgraph = addLayers(lgraph,hadamardProdLayer(Hi,'FiHi'));
            lgraph = connectLayers(lgraph,'Fr','FrHr');
            lgraph = connectLayers(lgraph,'Fi','FiHr');
            lgraph = connectLayers(lgraph,'Fr','FrHi');
            lgraph = connectLayers(lgraph,'Fi','FiHi');
            lgraph = addLayers(lgraph,subtractionLayer('Zr'));
            lgraph = addLayers(lgraph,additionLayer(2,'Name','Zi'));
            lgraph = connectLayers(lgraph,'FrHr','Zr/in1');
            lgraph = connectLayers(lgraph,'FiHi','Zr/in2');
            lgraph = connectLayers(lgraph,'FiHr','Zi/in1');
            lgraph = connectLayers(lgraph,'FrHi','Zi/in2');
            
            % ifft2 
            lgraph = addLayers(lgraph,ifft2DLayer('FZr'));
            lgraph = addLayers(lgraph,ifft2DLayer('FZi'));
            lgraph = connectLayers(lgraph,'Zr','FZr');
            lgraph = connectLayers(lgraph,'Zi','FZi');
            lgraph = addLayers(lgraph,subtractionLayer('Ur'));
            lgraph = addLayers(lgraph,additionLayer(2,'Name','Ui'));
            lgraph = connectLayers(lgraph,'FZr/real','Ur/in1');
            lgraph = connectLayers(lgraph,'FZi/imag','Ur/in2');
            lgraph = connectLayers(lgraph,'FZr/imag','Ui/in1');
            lgraph = connectLayers(lgraph,'FZi/real','Ui/in2');
            
            % intensity
            lgraph = addLayers(lgraph,intensityLayer('I'));
            lgraph = connectLayers(lgraph,'Ur','I/in1');
            lgraph = connectLayers(lgraph,'Ui','I/in2');
            
            % fftshift
%             lgraph = addLayers(lgraph,fftshiftLayer('fftshift2'));
%             lgraph = connectLayers(lgraph,'I','fftshift2');
            
            % crop
%             lgraph = addLayers(lgraph,crop2dLayer('centercrop','Name','crop'));
%             lgraph = connectLayers(lgraph,'fftshift2','crop/in');
%             lgraph = connectLayers(lgraph,'in','crop/ref');
            
            % Convert to dlnetwork.
            dlnet = dlnetwork(lgraph);
    
            % Set Network property.
            layer.Network = dlnet;
            
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            
            X = dlarray(X,'SSCB');
            
            % Predict using network.
            dlnet = layer.Network;
            Z = predict(dlnet,X);
            
            % Strip dimension labels.
            Z = stripdims(Z);
            
        end

    end
end