# Neural Networks Learning

In this exercise, you will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition. Unlike Excercise-3, here you will implement the algorithm to learn the parameters for the neural network.

Setup Neural network parameters:

    input_layer_size  = 400;  % 20x20 Input Images of Digits
    hidden_layer_size = 25;   % 25 hidden units
    num_labels = 10;          % 10 labels, from 1 to 10   
                              % (note that we have mapped "0" to label 10)

Loading and Visualising training data:

    load('ex4data1.mat');
    m = size(X, 1);
    sel = randperm(size(X, 1));
    sel = sel(1:100);
    displayData(X(sel, :));

For now, load some pre-initialized neural network parameters and unroll them:
    
    load('ex4weights.mat');
    nn_params = [Theta1(:) ; Theta2(:)];

It usually a good practice to compute Feedforward part that returns cost only. Also always first check the result without using regularisation and then check the result after implementing the regularisation.  
Computing Cost by implementing Feedforward part and regularisation:

    lambda = 1;
    J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
    
Now you will randomly initialise parameters and unroll them into one vector:

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

Implement the backpropagation algorithm for the neural network by writing the code in nnCostFunction() to return the partial derivatives of the parameters.Also Check gradients by running checkNNGradients

    checkNNGradients;
    lambda = 3;
    checkNNGradients(lambda);
    debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
    
To train your neural network, we will now use "fmincg", which is a function which works similarly to "fminunc". These advanced optimizers are able to train our cost functions efficiently as long as we provide them with the gradient computations.

    options = optimset('MaxIter', 50);
    lambda = 1;
    costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

Visualise your weights and compute the prediction accuracy:

    displayData(Theta1(:, 2:end));
    pred = predict(Theta1, Theta2, X);
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

## Functions used in above code

displayData(): Same as shown [here](https://github.com/sanket1012/Coursera-Machine-Learning/blob/master/Excercise%203-%20Multi-class%20Classification%20and%20Neural%20Networks/Logistic%20Regression/README.md)

nnCostFunction(): Including BP algorithm

    function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
        Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
        Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
        m = size(X, 1);
        J = 0;
        Theta1_grad = zeros(size(Theta1));
        Theta2_grad = zeros(size(Theta2));
        delta_1=zeros(size(Theta1));
        delta_2=zeros(size(Theta2));
        th1_1=size(Theta1,1);
        th1_2=size(Theta1,2);
        th2_1=size(Theta2,1);
        th2_2=size(Theta2,2);
        X = [ones(m, 1) X];
        h_theta_1=sigmoid(Theta1*X');         %25 x 5000
        h_theta_1 = [ones(1, m);h_theta_1];   %26 x 5000
        h_theta_2=sigmoid(Theta2*h_theta_1);  %10 x 5000
        Y=zeros(num_labels,m);
        for c=1:10
            for i=1:m
                if (y(i,1)==c)
                    Y(c,i)=1;
                end
            end
        end
        for i=1:m
            J_1=sum(-Y(:,i).*log(h_theta_2(:,i))-(1-Y(:,i)).*log(1-h_theta_2(:,i)));
            J=J+J_1;
        end

        J=(1/m)*J+(lambda/(2*m))*(sum(sum(Theta1(1:th1_1,2:th1_2).^2))+sum(sum(Theta2(1:th2_1,2:th2_2).^2)));
        for t=1:m
            a_1=X(t,:);
            a2=sigmoid(Theta1*a_1');         %25 x 1
            a_2 = [1;a2];                    %26 x 1
            a_3=sigmoid(Theta2*a_2);          %10 x 1
            for i=1:num_labels
                del_3(i,1)=(a_3(i,1)-Y(i,t));
            end
            z2=Theta1*a_1';
            z_2=[1;z2];
            g_dash=sigmoidGradient(z_2);
            del_2=(Theta2'*del_3).*g_dash;
            del_2=del_2(2:end);
            delta_1=delta_1+del_2*a_1;
            delta_2=delta_2+del_3*a_2';
        end
      
        Theta1_grad(:,1)=(1/m)*delta_1(1:th1_1,1);
        Theta1_grad(:,2:th1_2)=(1/m)*delta_1(1:th1_1,2:th1_2)+(lambda/m)*Theta1(1:th1_1,2:th1_2);
        Theta2_grad(:,1)=(1/m)*delta_2(1:th2_1,1);
        Theta2_grad(:,2:th2_2)=(1/m)*delta_2(1:th2_1,2:th2_2)+(lambda/m)*Theta2(1:th2_1,2:th2_2);

        grad = [Theta1_grad(:) ; Theta2_grad(:)];
    end

randInitializeWeights():

    function W = randInitializeWeights(L_in, L_out)
        W = zeros(L_out, 1 + L_in);
        epsilon_init = 0.12;
        W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
    end

checkNNGradients():

    function checkNNGradients(lambda)
        if ~exist('lambda', 'var') || isempty(lambda)
            lambda = 0;
        end

        input_layer_size = 3;
        hidden_layer_size = 5;
        num_labels = 3;
        m = 5;
        Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
        Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
        X  = debugInitializeWeights(m, input_layer_size - 1);
        y  = 1 + mod(1:m, num_labels)';
        nn_params = [Theta1(:) ; Theta2(:)];
        costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
        [cost, grad] = costFunc(nn_params);
        numgrad = computeNumericalGradient(costFunc, nn_params);
        disp([numgrad grad]);
        diff = norm(numgrad-grad)/norm(numgrad+grad);
    end
    
predict():

    function p = predict(Theta1, Theta2, X)
        m = size(X, 1);
        num_labels = size(Theta2, 1);
        p = zeros(size(X, 1), 1);
        h1 = sigmoid([ones(m, 1) X] * Theta1');
        h2 = sigmoid([ones(m, 1) h1] * Theta2');
        [dummy, p] = max(h2, [], 2);
    end

sigmoid():
    
    function g = sigmoid(z)
        g = 1.0 ./ (1.0 + exp(-z));
    end

sigmoidGradient():

    function g = sigmoidGradient(z)
        g = zeros(size(z));
        m=size(z);
        for i=1:m(1)
            for j=1:m(2)
                g1=sigmoid(z(i,j));
                g(i,j)=g1*(1-g1);
            end
        end
    end

computeNumericalGradient():

    function numgrad = computeNumericalGradient(J, theta)
        numgrad = zeros(size(theta));
        perturb = zeros(size(theta));
        e = 1e-4;
        for p = 1:numel(theta)
            % Set perturbation vector
            perturb(p) = e;
            loss1 = J(theta - perturb);
            loss2 = J(theta + perturb);
            % Compute Numerical Gradient
            numgrad(p) = (loss2 - loss1) / (2*e);
            perturb(p) = 0;
        end
    end
