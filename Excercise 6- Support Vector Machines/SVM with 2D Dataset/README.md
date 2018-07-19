# Support Vector Machines

Loading and Visualizing Data:

    load('ex6data1.mat');
    plotData(X, y);

Training Linear SVM and plot the decision boundary learned:

    load('ex6data1.mat');
    C = 1;
    model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
    visualizeBoundaryLinear(X, y, model);

Implementing Gaussian Kernel:

    x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
    sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :'\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n'], sigma, sim);

Visualizing Dataset 2:

    load('ex6data2.mat');
    plotData(X, y);

Training SVM with RBF Kernel (Dataset 2):

    load('ex6data2.mat');

    % SVM Parameters
    C = 1; sigma = 0.1;
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
    visualizeBoundary(X, y, model);

Visualizing Dataset 3:

    load('ex6data3.mat');

    % Plot training data
    plotData(X, y);

Training SVM with RBF Kernel (Dataset 3)

    load('ex6data3.mat');
    [C, sigma] = dataset3Params(X, y, Xval, yval,model);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    visualizeBoundary(X, y, model);
    
## Functions used in above code:

plotData():

    function plotData(X, y)
        pos = find(y == 1); neg = find(y == 0);
        plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7)
        hold on;
        plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7)
        hold off;
    end
    
visualizeBoundarylinear():

    function visualizeBoundaryLinear(X, y, model)
        w = model.w;
        b = model.b;
        xp = linspace(min(X(:,1)), max(X(:,1)), 100);
        yp = - (w(1)*xp + b)/w(2);
        plotData(X, y);
        hold on;
        plot(xp, yp, '-b'); 
        hold off
    end

gaussianKernel():

    function sim = gaussianKernel(x1, x2, sigma)
        x1 = x1(:); x2 = x2(:);
        sim = 0;
        for i=1:length(x1)
            for j=1:length(x1)
                sim=exp(-(norm(x1-x2).^2)/(2*sigma^2));
            end
        end
    end
