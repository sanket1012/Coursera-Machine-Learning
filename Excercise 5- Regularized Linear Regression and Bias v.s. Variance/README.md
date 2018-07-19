# Regularized Linear Regression and Bias v.s. Variance

In this part of Machine Learning, you will implement regularized linear regression and use it to study models with different bias-variance properties.

First, you will implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir.

Next, you will go through some diagnostics of debugging learning algorithms and examine the effects of bias v.s. variance

### Regularized Linear Regression

Loading and Visualizing Data:

    load ('ex5data1.mat');
    m = size(X, 1);
    plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
    xlabel('Change in water level (x)');
    ylabel('Water flowing out of the dam (y)');

Compute Cost and Gradient for Regularized Linear Regression:

    theta = [1 ; 1];
    J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
    
Training and visualise the Linear Regression:

    lambda = 0;
    [theta] = trainLinearReg([ones(m, 1) X], y, lambda);
    plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
    xlabel('Change in water level (x)');
    ylabel('Water flowing out of the dam (y)');
    hold on;
    plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
    hold off;

## Bias v.s. Variance

Implementing Learning Curve for Linear Regression

Since the model is underfitting the data, we expect to see a graph with "high bias"

    lambda = 0;
    [error_train, error_val] = learningCurve([ones(m, 1) X], y, [ones(size(Xval, 1), 1) Xval], yval, lambda);
    
    plot(1:m, error_train, 1:m, error_val);
    title('Learning curve for linear regression')
    legend('Train', 'Cross Validation')
    xlabel('Number of training examples')
    ylabel('Error')
    axis([0 13 0 150])
    for i = 1:m
        fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
    end
    
Feature Mapping for Polynomial Regression:

One solution to this is to use polynomial regression.

    p = 8;

    % Map X onto Polynomial Features and Normalize
    X_poly = polyFeatures(X, p);
    [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
    X_poly = [ones(m, 1), X_poly];                   % Add Ones

    % Map X_poly_test and normalize (using mu and sigma)
    X_poly_test = polyFeatures(Xtest, p);
    X_poly_test = bsxfun(@minus, X_poly_test, mu);
    X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
    X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

    % Map X_poly_val and normalize (using mu and sigma)
    X_poly_val = polyFeatures(Xval, p);
    X_poly_val = bsxfun(@minus, X_poly_val, mu);
    X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
    X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

Learning Curve for Polynomial Regression:  
The code below runs polynomial regression with lambda = 0. You should try running the code with different values of lambda to see how the fit and learning curve change.

    lambda = 0;
    [theta] = trainLinearReg(X_poly, y, lambda);

    % Plot training data and fit
    figure(1);
    plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
    plotFit(min(X), max(X), mu, sigma, theta, p);
    xlabel('Change in water level (x)');
    ylabel('Water flowing out of the dam (y)');
    title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

    figure(2);
    [error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
    plot(1:m, error_train, 1:m, error_val);

    title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
    xlabel('Number of training examples')
    ylabel('Error')
    axis([0 13 0 100])
    legend('Train', 'Cross Validation')

    fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
    fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
    for i = 1:m
        fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
    end

Validation for Selecting Lambda:

Implement validationCurve to test various values of lambda on a validation set. You will then use this to select the "best" lambda value.

    [lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval);
    close all;
    plot(lambda_vec, error_train, lambda_vec, error_val);
    legend('Train', 'Cross Validation');
    xlabel('lambda');
    ylabel('Error');

    fprintf('lambda\t\tTrain Error\tValidation Error\n');
    for i = 1:length(lambda_vec)
      fprintf(' %f\t%f\t%f\n', lambda_vec(i), error_train(i), error_val(i));
    end
    
## Functions used in the code above:

linearRegCostFunction():

    function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
        m = length(y); % number of training examples
        J = 0;
        grad = zeros(size(theta));

        h_theta=X*theta;

        J=(1/(2*m))*sum((h_theta-y).^2)+(lambda/(2*m))*sum(theta(2:end).^2);
        temp=theta;
        temp(1)=0;
        grad=(1/m)*X'*(h_theta-y)+(lambda/m)*temp;
        grad = grad(:);
    end

trainLinearReg():

    function [theta] = trainLinearReg(X, y, lambda)
        % Initialize Theta
        initial_theta = zeros(size(X, 2), 1); 

        % Create "short hand" for the cost function to be minimized
        costFunction = @(t) linearRegCostFunction(X, y, t, lambda);

        % Now, costFunction is a function that takes in only one argument
        options = optimset('MaxIter', 200, 'GradObj', 'on');

        % Minimize using fmincg
        theta = fmincg(costFunction, initial_theta, options);
    end

learningCurve():

    function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
        m = size(X, 1);
        error_train = zeros(m, 1);
        error_val   = zeros(m, 1);
        for i=1:m
            theta = trainLinearReg(X(1:i,:), y(1:i), lambda);
            Jtrain = linearRegCostFunction(X(1:i,:), y(1:i), theta, 0);
            Jval = linearRegCostFunction(Xval, yval, theta, 0);
            error_train(i)=Jtrain;
            error_val(i)=Jval;
        end
    end

polyFeatures():

    function [X_poly] = polyFeatures(X, p)
        X_poly = zeros(numel(X), p);
        for i=1:p
            X_poly(:,i)=X.^i;
        end
    end
    
featureNormalize():

    function [X_norm, mu, sigma] = featureNormalize(X)
        mu = mean(X);
        X_norm = bsxfun(@minus, X, mu);

        sigma = std(X_norm);
        X_norm = bsxfun(@rdivide, X_norm, sigma);
    end

plotFit():

    function plotFit(min_x, max_x, mu, sigma, theta, p)
        hold on;

        % We plot a range slightly bigger than the min and max values to get
        % an idea of how the fit will vary outside the range of the data points
        x = (min_x - 15: 0.05 : max_x + 25)';

        % Map the X values 
        X_poly = polyFeatures(x, p);
        X_poly = bsxfun(@minus, X_poly, mu);
        X_poly = bsxfun(@rdivide, X_poly, sigma);

        % Add ones
        X_poly = [ones(size(x, 1), 1) X_poly];

        % Plot
        plot(x, X_poly * theta, '--', 'LineWidth', 2)

        % Hold off to the current figure
        hold off
    end
    
    
