# Anomaly Detection System

We will implement an anomaly detection algorithm to detect anomalous behavior in server computers. The features measure the throughput (mb/s) and latency (ms) of response of each server. While your servers were operating, you collected m = 307 examples of how they were behaving, and thus have an unlabeled dataset fx(1); : : : ; x(m)g. You suspect that the vast majority of these examples are "normal" (non-anomalous) examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset.  
We will use a Gaussian model to detect anomalous examples in the dataset. First start on a 2D dataset that will allow you to visualize
what the algorithm is doing. On that dataset you will fit a Gaussian distribution and then find values that have very low probability and hence can be considered anomalies. After that, you will apply the anomaly detection algorithm to a larger dataset with many dimensions.

Load Example Datase:

    load('ex8data1.mat');
    %  Visualize the example dataset
    plot(X(:, 1), X(:, 2), 'bx');
    axis([0 30 0 30]);
    xlabel('Latency (ms)');
    ylabel('Throughput (mb/s)');

Estimate the dataset statistics:

    %  Estimate my and sigma2
    [mu sigma2] = estimateGaussian(X);

    %  Returns the density of the multivariate normal at each data point (row) 
    %  of X
    p = multivariateGaussian(X, mu, sigma2);

    %  Visualize the fit
    visualizeFit(X,  mu, sigma2);
    xlabel('Latency (ms)');
    ylabel('Throughput (mb/s)');

Find Outliers, find a good epsilon threshold using a cross-validation set probabilities given the estimated Gaussian distribution

    pval = multivariateGaussian(Xval, mu, sigma2);

    [epsilon F1] = selectThreshold(yval, pval);
    fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
    fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
    fprintf('   (you should see a value epsilon of about 8.99e-05)\n');
    fprintf('   (you should see a Best F1 value of  0.875000)\n\n');

    %  Find the outliers in the training set and plot the
    outliers = find(p < epsilon);

    %  Draw a red circle around those outliers
    hold on
    plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
    hold off

Multidimensional Outliers, using above code apply it to the harder problem in which more features describe each datapoint and only some features indicate whether a point is an outlier.

    load('ex8data2.mat');
    %  Apply the same steps to the larger dataset
    [mu sigma2] = estimateGaussian(X);

    %  Training set 
    p = multivariateGaussian(X, mu, sigma2);

    %  Cross-validation set
    pval = multivariateGaussian(Xval, mu, sigma2);

    %  Find the best threshold
    [epsilon F1] = selectThreshold(yval, pval);

    fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
    fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
    fprintf('   (you should see a value epsilon of about 1.38e-18)\n');
    fprintf('   (you should see a Best F1 value of 0.615385)\n');
    fprintf('# Outliers found: %d\n\n', sum(p < epsilon));

## Functioned used in above code:

estimateGaussian():

    function [mu sigma2] = estimateGaussian(X)
        [m, n] = size(X);
        mu = zeros(n, 1);
        sigma2 = zeros(n, 1);
        for i=1:n
            mu(i,1)=(1/m)*sum(X(:,i));
        end

        for i=1:n
            sigma2(i,1)=(1/m)*sum((X(:,i)-mu(i,1)).^2);
        end
    end

multivariateGaussian():

    function p = multivariateGaussian(X, mu, Sigma2)
        k = length(mu);

        if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
            Sigma2 = diag(Sigma2);
        end

        X = bsxfun(@minus, X, mu(:)');
        p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));
    end
    
selectThreshold():

    function [bestEpsilon bestF1] = selectThreshold(yval, pval)
        bestEpsilon = 0;
        bestF1 = 0;
        F1 = 0;

        stepsize = (max(pval) - min(pval)) / 1000;
        for epsilon = min(pval):stepsize:max(pval)
            cvPredictions=pval<epsilon;
            fp = sum((cvPredictions == 1) &(yval == 0));
            tp= sum((cvPredictions == 1) &(yval == 1));
            fn= sum((cvPredictions == 0) &(yval == 1));

            prec =tp/(tp + fp);
            rec =tp/(tp + fn);

            F1=2*prec*rec/(prec + rec);
            if F1 > bestF1
               bestF1 = F1;
               bestEpsilon = epsilon;
            end
        end
    end
