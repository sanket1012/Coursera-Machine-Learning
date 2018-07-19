# SVM with Spam Classifier

Email Preprocessing, to use an SVM to classify emails into Spam v.s. Non-Spam, we first need to convert each email into a vector of features.

Implement the preprocessing steps for each email to produce a word indices vector for a given email.

    % Extract Features
    file_contents = readFile('emailSample1.txt');
    word_indices  = processEmail(file_contents);

    % Print Stats
    fprintf('Word Indices: \n');
    fprintf(' %d', word_indices);
    fprintf('\n\n');

Feature Extraction, convert each email into a vector of features in R^n:

    file_contents = readFile('emailSample1.txt');
    word_indices  = processEmail(file_contents);
    features      = emailFeatures(word_indices);

    % Print Stats
    fprintf('Length of feature vector: %d\n', length(features));
    fprintf('Number of non-zero entries: %d\n', sum(features > 0));

Train Linear SVM for Spam Classification:

    load('spamTrain.mat');
    fprintf('\nTraining Linear SVM (Spam Classification)\n')
    fprintf('(this may take 1 to 2 minutes) ...\n')
    C = 0.1;
    model = svmTrain(X, y, C, @linearKernel);
    p = svmPredict(model, X);
    fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

Test Spam Classification:

    load('spamTest.mat');
    fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')
    p = svmPredict(model, Xtest);
    fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);

Top Predictors of Spam: Since the model we are training is a linear SVM, we can inspect the weights learned by the model to understand better how it is determining whether an email is spam or not. The following code finds the words with the highest weights in the classifier. Informally, the classifier 'thinks' that these words are the most likely indicators of spam.

Sort the weights and obtain the vocabulary list:

    [weight, idx] = sort(model.w, 'descend');
    vocabList = getVocabList();
    fprintf('\nTop predictors of spam: \n');
    for i = 1:15
        fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
    end
    
## Functions used in above code:

readFile():

    function file_contents = readFile(filename)
        fid = fopen(filename);
        if fid
            file_contents = fscanf(fid, '%c', inf);
            fclose(fid);
        else
            file_contents = '';
            fprintf('Unable to open %s\n', filename);
        end
    end
    
processEmail():

    function word_indices = processEmail(email_contents)
        vocabList = getVocabList();
        word_indices = [];
        email_contents = lower(email_contents);
        email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

        % Handle Numbers
        % Look for one or more characters between 0-9
        email_contents = regexprep(email_contents, '[0-9]+', 'number');

        % Handle URLS
        % Look for strings starting with http:// or https://
        email_contents = regexprep(email_contents, ...
                                   '(http|https)://[^\s]*', 'httpaddr');

        % Handle Email Addresses
        % Look for strings with @ in the middle
        email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

        % Handle $ sign
        email_contents = regexprep(email_contents, '[$]+', 'dollar');

        % Process file
        l = 0;

        while ~isempty(email_contents)

            % Tokenize and also get rid of any punctuation
            [str, email_contents] = strtok(email_contents, [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);

            % Remove any non alphanumeric characters
            str = regexprep(str, '[^a-zA-Z0-9]', '');

            % Stem the word 
            % (the porterStemmer sometimes has issues, so we use a try catch block)
            try str = porterStemmer(strtrim(str)); 
            catch str = ''; continue;
            end;

            % Skip the word if it is too short
            if length(str) < 1
               continue;
            end
            for i=1:length(vocabList)
                if (strcmp(vocabList(i),str))
                    word_indices=[word_indices;i];
                end
            end
            if (l + length(str) + 1) > 78
                fprintf('\n');
                l = 0;
            end
            fprintf('%s ', str);
            l = l + length(str) + 1;
        end
    end

emailFeatures():

    function x = emailFeatures(word_indices)
        n = 1899;
        x = zeros(n, 1);
        for i=1:length(word_indices)
            x(word_indices(i))=1;
        end
    end
    
getVocabList():

    Function vocabList = getVocabList()
        %% Read the fixed vocabulary list
        fid = fopen('vocab.txt');

        % Store all dictionary words in cell array vocab{}
        n = 1899;  % Total number of words in the dictionary

        vocabList = cell(n, 1);
        for i = 1:n
            % Word Index (can ignore since it will be = i)
            fscanf(fid, '%d', 1);
            % Actual Word
            vocabList{i} = fscanf(fid, '%s', 1);
        end
        fclose(fid);
    end
    
