# Logistic Regression Without Regularisation

In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university. Suppose that you are the administrator of a university department and you want to determine each applicant's chance of admission based on their results on two exams. You have historical data [**ex2data1.txt**](https://github.com/sanket1012/Coursera-Machine-Learning/blob/master/Excercise%202-%20Logistic%20Regression/ex2data1.txt) from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant's scores on two exams and the admissions decision.

Your task is to build a classification model that estimates an applicant's probability of admission based the scores from those two exams.

First lets visualise the data:

    data = load('ex2data1.txt');
    X = data(:, [1, 2]); y = data(:, 3);
    fprintf(['Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n']);
    plotData(X, y);
    hold on;
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend('Admitted', 'Not admitted')
    hold off;

Initialization of few paramaters:

    [m, n] = size(X);
    X = [ones(m, 1) X];
    initial_theta = zeros(n + 1, 1);

Compute Cost and Gradient Update by using optimization tool (fminunc):

    [cost, grad] = costFunction(initial_theta, X, y);
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    [theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
    
Plotting the obtained decision boundary:

    plotDecisionBoundary(theta, X, y);
    hold on;
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend('Admitted', 'Not admitted')
    hold off;
    
Predict probability and computer the accuracy of the prediction for a student with score 45 on exam 1 and score 85 on exam 2:

    prob = sigmoid([1 45 85] * theta);
    fprintf(['For a student with scores 45 and 85, we predict an admission probability of %f\n'], prob);
    fprintf('Expected value: 0.775 +/- 0.002\n\n');
    p = predict(theta, X);
    fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
    fprintf('Expected accuracy (approx): 89.0\n');
    fprintf('\n');

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
