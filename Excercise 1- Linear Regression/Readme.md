# Linear Regression
The file [ex1data1.txt](https://github.com/sanket1012/Coursera-Machine-Learning/blob/master/Excercise%201-%20Linear%20Regression/ex1data1.txt) contains the dataset for our linear regression problem. The first column is the population of a city and the second column is the profit of a food truck in that city. A negative value for profit indicates a loss.

First Load the data and plot it to have a visualisation abaout the data:
    
    data = load('ex1data1.txt');  
    X = data(:, 1); y = data(:, 2);
    plotData(X, y);
    
Perform initialization and seeting up your training data:
    
    m = length(y);
    X = [ones(m, 1), data(:,1)];      % Add a column of ones to x
    theta = zeros(2, 1);              % initialize fitting parameters
    
    iterations = 1500;                % Some gradient descent settings
    alpha = 0.01;
    
Compute the cost and gradient update:

    J = computeCost(X, y, theta);
    theta = gradientDescent(X, y, theta, alpha, iterations);

Plot the linear fit:

    plot(X(:,2), X*theta, '-')
    legend('Training data', 'Linear regression')
    
Predict values for population sizes of 35,000 and 70,000:
    
    predict1 = [1, 3.5] *theta;
    fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);
    predict2 = [1, 7] * theta;
    fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);
    
Visualize Cost Function:

    theta0_vals = linspace(-10, 10, 100);
    theta1_vals = linspace(-1, 4, 100);

    J_vals = zeros(length(theta0_vals), length(theta1_vals));     % initialize J_vals to a matrix of 0's
    for i = 1:length(theta0_vals)
        for j = 1:length(theta1_vals)
	        t = [theta0_vals(i); theta1_vals(j)];
	        J_vals(i,j) = computeCost(X, y, t);
        end
    end

    J_vals = J_vals';
    figure;
    surf(theta0_vals, theta1_vals, J_vals)
    xlabel('\theta_0'); ylabel('\theta_1');

Implementing a contour plot for Cost Function:

    figure;
    contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    xlabel('\theta_0'); ylabel('\theta_1');
    hold on;
    plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    
##  Functionsused in Linear Regression:
 
 computeCost():
 
    function J = computeCost(X, y, theta)
      m = length(y);    % number of training examples
      alpha=0.01;
      J = 0;

      h_theta=X*theta;
      J=sum((1/(2*m))*(h_theta-y).^2);
    end
    
gradientDescent():
   
    function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
      m = length(y);      % number of training examples
      J_history = zeros(num_iters, 1);

      for iter = 1:num_iters
        h_theta=X*theta;
        for i=1:length(X)
          theta=theta-(alpha/m)*(h_theta(i,1)-y(i,1))*X(i,:)';
        end
        J_history(iter) = computeCost(X, y, theta);
      end
    end
    
plotData():

    function plotData(x, y)
      figure; % open a new figure window
      plot(x, y, 'rx', 'MarkerSize', 10); % Plot the data
      ylabel('Profit in $10,000s'); % Set the y?axis label
      xlabel('Population of City in 10,000s'); % Set the x?axis label
    end
