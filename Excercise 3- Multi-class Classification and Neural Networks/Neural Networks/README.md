# Multi Class Classification using Neural Networks

Setup your Neural Network model:

    input_layer_size  = 400;  % 20x20 Input Images of Digits
    hidden_layer_size = 25;   % 25 hidden units
    num_labels = 10;          % 10 labels, from 1 to 10   
                              % (note that we have mapped "0" to label 10)

Loading and visualising data:

    load('ex3data1.mat');
    m = size(X, 1);
    % Randomly select 100 data points to display
    sel = randperm(size(X, 1));
    sel = sel(1:100);
    displayData(X(sel, :));
    
Here parameters are already saved from trained neural networks. Loading these parameters: (you can find these parameters [here](https://github.com/sanket1012/Coursera-Machine-Learning/blob/master/Excercise%203-%20Multi-class%20Classification%20and%20Neural%20Networks/ex3weights.mat)
    
    load('ex3weights.mat');
    
Predict the label and compute the accuracy:

    pred = predict(Theta1, Theta2, X);
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
    rp = randperm(m);
    for i = 1:m
      % Display 
      fprintf('\nDisplaying Example Image\n');
      displayData(X(rp(i), :));
      pred = predict(Theta1, Theta2, X(rp(i),:));
      fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    end
    
## Functions used in above code

displayData(): Same as shown [here](https://github.com/sanket1012/Coursera-Machine-Learning/blob/master/Excercise%203-%20Multi-class%20Classification%20and%20Neural%20Networks/Logistic%20Regression/README.md)

predict():

    function p = predict(Theta1, Theta2, X)
      m = size(X, 1);
      num_labels = size(Theta2, 1);
      p = zeros(size(X, 1), 1);
      X = [ones(m, 1) X];
      h1=sigmoid(X*Theta1');
      h1=[ones(m,1) h1];
      h2=sigmoid(h1*Theta2');

      [temp,p]=max(h2,[],2);
    end
    
