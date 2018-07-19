
# K-Means Clustering

To implement K-Means, we will perform two steps in the learning algorithm; findClosestCentroids and computeCentroids.

Find Closest Centroids:
    
    load('ex7data2.mat');
    % Select an initial set of centroids
    K = 3; % 3 Centroids
    initial_centroids = [3 3; 6 2; 8 5];

Find the closest centroids for the examples using the `initial_centroids`:

    idx = findClosestCentroids(X, initial_centroids);
    fprintf('Closest centroids for the first 3 examples: \n')
    fprintf(' %d', idx(1:3));
    fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');

Compute Mean, after implementing the closest centroids function:
    
    centroids = computeCentroids(X, idx, K);
    fprintf('Centroids computed after initial finding of closest centroids: \n')
    fprintf(' %f %f \n' , centroids');
    fprintf('\n(the centroids should be\n');
    fprintf('   [ 2.428301 3.157924 ]\n');
    fprintf('   [ 5.813503 2.633656 ]\n');
    fprintf('   [ 7.119387 3.616684 ]\n\n');

K-Means Clustering after you have completed the two functions computeCentroids and findClosestCentroids, you have all the necessary pieces to run the kMeans algorithm. Now lets implement the K-Means algorithm on the example dataset provided:

    load('ex7data2.mat');

    % Settings for running K-Means
    K = 3;
    max_iters = 10;

For consistency, here we set centroids to specific values but in practice you want to generate them automatically, such as by settings them to be random examples (as can be seen in kMeansInitCentroids):

    initial_centroids = [3 3; 6 2; 8 5];
    % Run K-Means algorithm. The 'true' at the end tells our function to plot the progress of K-Means
    [centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
    fprintf('\nK-Means Done.\n\n');

K-Means Clustering on Pixels, use K-Means to compress an image. To do this, we will first run K-Means on the colors of the pixels in the image and then you will map each pixel onto its closest centroid.

    A = double(imread('bird_small.png'));
    % If imread does not work for you, you can try instead
    % load ('bird_small.mat');
    A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

    % Size of the image
    img_size = size(A);

    % Reshape the image into an Nx3 matrix where N = number of pixels.
    % Each row will contain the Red, Green and Blue pixel values
    % This gives us our dataset matrix X that we will use K-Means on.
    X = reshape(A, img_size(1) * img_size(2), 3);

    % Run your K-Means algorithm on this data, you should try different values of K and max_iters here
    K = 16; 
    max_iters = 10;

    % When using K-Means, it is important the initialize the centroids randomly. 
    % You should complete the code in kMeansInitCentroids.m before proceeding
    initial_centroids = kMeansInitCentroids(X, K);

    % Run K-Means
    [centroids, idx] = runkMeans(X, initial_centroids, max_iters);

Image Compression, we will use the clusters of K-Means to compress an image. To do this, we first find the closest clusters for each example.

    idx = findClosestCentroids(X, centroids);

    % Essentially, now we have represented the image X as in terms of the indices in idx. 

    % We can now recover the image from the indices (idx) by mapping each pixel (specified by its index in idx) to the centroid value
    X_recovered = centroids(idx,:);

    % Reshape the recovered image into proper dimensions
    X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

    % Display the original image 
    subplot(1, 2, 1);
    imagesc(A); 
    title('Original');

    % Display compressed image side by side
    subplot(1, 2, 2);
    imagesc(X_recovered)
    title(sprintf('Compressed, with %d colors.', K));

## Functions used in above code:

findClosestCentroids():

    function idx = findClosestCentroids(X, centroids)
        m=size(X,1);
        K = size(centroids, 1);
        temp=[];
        idx = zeros(size(X,1), 1);
        for i=1:m
            for j=1:K
                temp(j,:)=norm(X(i,:)-centroids(j,:))^2;
            end
            [~,Index]=min(temp);
            idx(i,1)=Index;
        end

    end
    
computeCentroids():

    function centroids = computeCentroids(X, idx, K)
        [m n] = size(X);
        centroids = zeros(K, n);
        for j=1:K
            temp=zeros(1,n);
            temp1=0;
            for p=1:m
                if (idx(p,1)==j)
                    temp=temp+X(p,:);
                    temp1=temp1+1;
                end
            end
            sum(j,:)=temp;
            centroids(j,:)=sum(j,:)/temp1;
        end
    end
    
runKmeans():

    function [centroids, idx] = runkMeans(X, initial_centroids, max_iters, plot_progress)
        if ~exist('plot_progress', 'var') || isempty(plot_progress)
            plot_progress = false;
        end

        % Plot the data if we are plotting progress
        if plot_progress
            figure;
            hold on;
        end

        % Initialize values
        [m n] = size(X);
        K = size(initial_centroids, 1);
        centroids = initial_centroids;
        previous_centroids = centroids;
        idx = zeros(m, 1);

        % Run K-Means
        for i=1:max_iters

            % Output progress
            fprintf('K-Means iteration %d/%d...\n', i, max_iters);
            if exist('OCTAVE_VERSION')
                fflush(stdout);
            end

            % For each example in X, assign it to the closest centroid
            idx = findClosestCentroids(X, centroids);

            % Optionally, plot progress here
            if plot_progress
                plotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
                previous_centroids = centroids;
                fprintf('Press enter to continue.\n');
                pause;
            end

            % Given the memberships, compute new centroids
            centroids = computeCentroids(X, idx, K);
        end

        % Hold off if we are plotting progress
        if plot_progress
            hold off;
        end
    end

kMeansInitCentroids():

    function centroids = kMeansInitCentroids(X, K)
        centroids = zeros(K, size(X, 2));
        randidx = randperm(size(X, 1));
        % Take the first K examples as centroids
        centroids = X(randidx(1:K), :);
    end

