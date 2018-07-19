# Recommender System

We will implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings. This dataset consists of ratings on a scale of 1 to 5. The dataset has nu = 943 users, and nm = 1682 movies. Also we will implement the function to compute the collaborative fitlering objective function and gradient. After implementing the cost function and gradient, you will use fmincg.m to
learn the parameters for collaborative filtering.

Loading movie ratings dataset:

    load ('ex8_movies.mat');
    %  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
    %  943 users
    %
    %  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
    %  rating to movie i

    %  From the matrix, we can compute statistics like average rating.
    fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', mean(Y(1, R(1, :))));

    %  We can "visualize" the ratings matrix by plotting it with imagesc
    imagesc(Y);
    ylabel('Movies');
    xlabel('Users');

Collaborative Filtering Cost Function:

    %  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    load ('ex8_movieParams.mat');

    %  Reduce the data set size so that this runs faster
    num_users = 4; num_movies = 5; num_features = 3;
    X = X(1:num_movies, 1:num_features);
    Theta = Theta(1:num_users, 1:num_features);
    Y = Y(1:num_movies, 1:num_users);
    R = R(1:num_movies, 1:num_users);

    %  Evaluate cost function
    J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, num_features, 0);

    fprintf(['Cost at loaded parameters: %f \n(this value should be about 22.22)\n'], J);

Collaborative Filtering Gradient:

    %  Check gradients by running checkNNGradients
    checkCostFunction;

Collaborative Filtering Cost Regularization:

    %  Evaluate cost function
    J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, num_features, 1.5);
    fprintf(['Cost at loaded parameters (lambda = 1.5): %f \n(this value should be about 31.34)\n'], J);

Collaborative Filtering Gradient Regularization:

    %  Check gradients by running checkNNGradients
    checkCostFunction(1.5);

Entering ratings for a new user:

    movieList = loadMovieList();

    %  Initialize my ratings
    my_ratings = zeros(1682, 1);

    % Check the file movie_idx.txt for id of each movie in our dataset
    % For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
    my_ratings(1) = 4;

    % Or suppose did not enjoy Silence of the Lambs (1991), you can set
    my_ratings(98) = 2;

    % We have selected a few movies we liked / did not like and the ratings we
    % gave are as follows:
    my_ratings(7) = 3;
    my_ratings(12)= 5;
    my_ratings(54) = 4;
    my_ratings(64)= 5;
    my_ratings(66)= 3;
    my_ratings(69) = 5;
    my_ratings(183) = 4;
    my_ratings(226) = 5;
    my_ratings(355)= 5;

    fprintf('\n\nNew user ratings:\n');
    for i = 1:length(my_ratings)
        if my_ratings(i) > 0 
            fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
        end
    end

Learning Movie Ratings:

    load('ex8_movies.mat');

    %  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
    %  943 users
    %
    %  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
    %  rating to movie i

    %  Add our own ratings to the data matrix
    Y = [my_ratings Y];
    R = [(my_ratings ~= 0) R];

    %  Normalize Ratings
    [Ynorm, Ymean] = normalizeRatings(Y, R);

    %  Useful Values
    num_users = size(Y, 2);
    num_movies = size(Y, 1);
    num_features = 10;

    % Set Initial Parameters (Theta, X)
    X = randn(num_movies, num_features);
    Theta = randn(num_users, num_features);

    initial_parameters = [X(:); Theta(:)];

    % Set options for fmincg
    options = optimset('GradObj', 'on', 'MaxIter', 100);

    % Set Regularization
    lambda = 10;
    theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features, lambda)), initial_parameters, options);

    % Unfold the returned theta back into U and W
    X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
    Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

    fprintf('Recommender system learning completed.\n');

Recommendation for you: After training the model, you can now make recommendations by computing the predictions matrix.

    p = X * Theta';
    my_predictions = p(:,1) + Ymean;

    movieList = loadMovieList();

    [r, ix] = sort(my_predictions, 'descend');
    fprintf('\nTop recommendations for you:\n');
    for i=1:10
        j = ix(i);
        fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
                movieList{j});
    end

    fprintf('\n\nOriginal ratings provided:\n');
    for i = 1:length(my_ratings)
        if my_ratings(i) > 0 
            fprintf('Rated %d for %s\n', my_ratings(i), ...
                     movieList{i});
        end
    end
    
## Functions used in above code:

cofiCostFunc():

    function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
        % Unfold the U and W matrices from params
        X = reshape(params(1:num_movies*num_features), num_movies, num_features);
        Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);
        J = 0;
        X_grad = zeros(size(X));
        Theta_grad = zeros(size(Theta));

        % Notes: X - num_movies  x num_features matrix of movie features
        %        Theta - num_users  x num_features matrix of user features
        %        Y - num_movies x num_users matrix of user ratings of movies
        %        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
        %            i-th movie was rated by the j-th user

        for i=1:num_movies
            for j=1:num_users
                if (R(i,j)==1)
                    J=J+(X(i,:)*Theta(j,:)'-Y(i,j))^2;
                    X_grad(i,:)=X_grad(i,:)+(X(i,:)*Theta(j,:)'-Y(i,j))*Theta(j,:);
                    Theta_grad(j,:)=Theta_grad(j,:)+(X(i,:)*Theta(j,:)'-Y(i,j))*X(i,:);
                end
            end
        end
        J=1/2*J;
        reg1=0;
        reg1_grad=zeros(size(Theta));
        reg2=0;
        reg2_grad=zeros(size(X));
        for i=1:num_features
            for j=1:num_users
                reg1=reg1+Theta(j,i)^2;
                reg1_grad(j,:)=lambda*Theta(j,:);
            end
            for j=1:num_movies
                reg2=reg2+X(j,i)^2;
                reg2_grad(j,:)=lambda*X(j,:);
            end
        end
        J=J+(lambda/2)*(reg1+reg2);
        Theta_grad=Theta_grad+reg1_grad;
        X_grad=X_grad+reg2_grad;
        grad = [X_grad(:); Theta_grad(:)];
    end
