# Principle Component Analysis

We start by using a small dataset that is easy to visualize

    load ('ex7data1.mat');
    plot(X(:, 1), X(:, 2), 'bo');
    axis([0.5 6.5 2 8]); axis square;

Implementing Principal Component Analysis, a dimension reduction technique:

    %  Before running PCA, it is important to first normalize X
    [X_norm, mu, sigma] = featureNormalize(X);

    %  Run PCA
    [U, S] = pca(X_norm);

    %  Compute mu, the mean of the each feature
    %  Draw the eigenvectors centered at mean of data. These lines show the directions of maximum variations in the dataset.
    hold on;
    drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
    drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
    hold off;

    fprintf('Top eigenvector: \n');
    fprintf(' U(:,1) = %f %f \n', U(1,1), U(2,1));
    fprintf('\n(you should expect to see -0.707107 -0.707107)\n');

Dimension Reduction, implementing the projection step to map the data onto the first k eigenvectors. The code will then plot the data in this reduced dimensional space. 
This will show you what the data looks like when using only the corresponding eigenvectors to reconstruct it.

    plot(X_norm(:, 1), X_norm(:, 2), 'bo');
    axis([-4 3 -4 3]); axis square

    %  Project the data onto K = 1 dimension
    K = 1;
    Z = projectData(X_norm, U, K);
    fprintf('Projection of the first example: %f\n', Z(1));
    fprintf('\n(this value should be about 1.481274)\n\n');

    X_rec  = recoverData(Z, U, K);
    fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));
    fprintf('\n(this value should be about  -1.047419 -1.047419)\n\n');

    %  Draw lines connecting the projected points to the original points
    hold on;
    plot(X_rec(:, 1), X_rec(:, 2), 'ro');
    for i = 1:size(X_norm, 1)
        drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
    end
    hold off

Loading and Visualizing Face Data:

    load ('ex7faces.mat')
    displayData(X(1:100, :));

PCA on Face Data: Eigenfaces, run PCA and visualize the eigenvectors which are in this case eigenfaces We display the first 36 eigenfaces.

    [X_norm, mu, sigma] = featureNormalize(X);
    %  Run PCA
    [U, S] = pca(X_norm);

    %  Visualize the top 36 eigenvectors found
    displayData(U(:, 1:36)');

Dimension Reduction for Faces, project images to the eigen space using the top k eigenvectors 

    K = 100;
    Z = projectData(X_norm, U, K);

    fprintf('The projected data Z has a size of: ')
    fprintf('%d ', size(Z));

Visualization of Faces after PCA Dimension Reduction, project images to the eigen space using the top K eigen vectors and visualize only using those K dimensions. 
Compare to the original input, which is also displayed:

    K = 100;
    X_rec  = recoverData(Z, U, K);

    % Display normalized data
    subplot(1, 2, 1);
    displayData(X_norm(1:100,:));
    title('Original faces');
    axis square;

    % Display reconstructed data from only k eigenfaces
    subplot(1, 2, 2);
    displayData(X_rec(1:100,:));
    title('Recovered faces');
    axis square;
