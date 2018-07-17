# Logistic Regression Without Regularisation

In this part of the Machine Learning, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly. Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected. To help you make the decision, you have a dataset [**ex2data2.txt**]() of test results on past microchips, from which you can build a logistic regression model.

Your task is to build a classification model that estimates an applicant's probability of admission based the scores from those two exams.

First lets visualise the data:

    data = load('ex2data2.txt');
    X = data(:, [1, 2]); y = data(:, 3);
    plotData(X, y);
    hold on;
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend('y = 1', 'y = 0')
    hold off;

Initialization of few paramaters:

    X = mapFeature(X(:,1), X(:,2));
    initial_theta = zeros(size(X, 2), 1);
    lambda = 1;

Compute Cost and Gradient Update by using optimization tool (fminunc):

    [cost, grad] = costFunctionReg(initial_theta, X, y, lambda);
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
        
Plotting the obtained decision boundary:

    title(sprintf('lambda = %g', lambda))
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend('y = 1', 'y = 0', 'Decision boundary')
    hold off;
    
Predict probability and computer the accuracy:

    p = predict(theta, X);
    fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
    fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');

##  Functionsused in Linear Regression:

costFunction():

    function [J, grad] = costFunction(theta, X, y)
      m = length(y); % number of training examples
      J = 0;
      grad = zeros(size(theta));
    =================================================================
    
    ==================================================================
    
    end
    
plotData():

    function plotData(X, y)
      figure; hold on;
=========================================================================

==========================================================================

      hold off;

    end
    
plotDecisionBoundary():

    function plotDecisionBoundary(theta, X, y)
      plotData(X(:,2:3), y);
      hold on
      if size(X, 2) <= 3
        plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
        plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
        plot(plot_x, plot_y)
        legend('Admitted', 'Not admitted', 'Decision Boundary')
        axis([30, 100, 30, 100])
      else
        u = linspace(-1, 1.5, 50);
        v = linspace(-1, 1.5, 50);
        z = zeros(length(u), length(v));
        for i = 1:length(u)
          for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
          end
        end
        z = z'; % important to transpose z before calling contour
        contour(u, v, z, [0, 0], 'LineWidth', 2)
      end
      hold off
    end
    
predict():

    function p = predict(theta, X)
      m = size(X, 1); % Number of training examples
      p = zeros(m, 1);
      
    end
