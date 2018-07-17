# Multi Class Classification using Logistic regression

Load and visualise the data:

    load('ex3data1.mat'); % training data stored in arrays X, y
    m = size(X, 1);
    rand_indices = randperm(m);
    sel = X(rand_indices(1:100), :);
    displayData(sel);

Setup the parameters:

    input_layer_size  = 400;  % 20x20 Input Images of Digits
    num_labels = 10;          % 10 labels, from 1 to 10
                              % (note that we have mapped "0" to label 10)

One Vs All training:

    lambda = 0.1;
    [all_theta] = oneVsAll(X, y, num_labels, lambda);

Predicting for One Vs All:

    pred = predictOneVsAll(all_theta, X);
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

## Functions used in above code

display Data():

    function [h, display_array] = displayData(X, example_width)
      if ~exist('example_width', 'var') || isempty(example_width) 
	      example_width = round(sqrt(size(X, 2)));
      end
      colormap(gray);
      
      % Compute rows, cols
      [m n] = size(X);
      example_height = (n / example_width);

      % Compute number of items to display
      display_rows = floor(sqrt(m));
      display_cols = ceil(m / display_rows);

      % Between images padding
      pad = 1;

      % Setup blank display
      display_array = - ones(pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad));

      % Copy each example into a patch on the display array
      curr_ex = 1;
      for j = 1:display_rows
	      for i = 1:display_cols
		      if curr_ex > m, 
			      break; 
		      end
		      % Copy the patch	
		      % Get the max value of the patch
		      max_val = max(abs(X(curr_ex, :)));
		      display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), pad + (i - 1) * (example_width + pad) +
          (1:example_width)) = reshape(X(curr_ex, :), example_height, example_width) / max_val;
		      curr_ex = curr_ex + 1;
	      end
	      if curr_ex > m, 
		      break; 
	      end
      end

      % Display Image
      h = imagesc(display_array, [-1 1]);

      % Do not show axis
      axis image off

      drawnow;

    end

OneVsAll():

    function [all_theta] = oneVsAll(X, y, num_labels, lambda)
      m = size(X, 1);
      n = size(X, 2);
      all_theta = zeros(num_labels, n + 1);
      X = [ones(m, 1) X];
      initial_theta = zeros(n + 1, 1);
      options = optimset('GradObj', 'on', 'MaxIter', 50); 
      for c=1:num_labels
        [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)),initial_theta, options);
        all_theta(c,:)=theta;
      end
    end

lrCostFunction():

    function [J, grad] = lrCostFunction(theta, X, y, lambda)
      m = length(y); % number of training examples
      J = 0;
      grad = zeros(size(theta));
      h_theta=sigmoid(X*theta);
      temp=ones(m,1);

      J=(1/m)*sum((-y.*log(h_theta))-((temp-y).*log(temp-h_theta)))+(lambda/2)*(1/m)*sum(theta(2:length(theta)).^2);

      grad= (1/m)*(X'*(h_theta-y));
      temp=theta;
      temp(1)=0;
      grad=grad+(lambda/m)*temp;
      grad = grad(:);
    end

sigmoid():

    function g = sigmoid(z)
      g = 1.0 ./ (1.0 + exp(-z));
    end

predictOneVsAll():

    function p = predictOneVsAll(all_theta, X)
      m = size(X, 1);
      num_labels = size(all_theta, 1);

      % You need to return the following variables correctly 
      p = zeros(size(X, 1), 1);

      % Add ones to the X data matrix
      X = [ones(m, 1) X];
      h=sigmoid(X*all_theta');
      [temp,p]=max(h,[],2);
    end
    
