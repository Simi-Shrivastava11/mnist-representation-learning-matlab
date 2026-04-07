%% DATA 604 - Final Project
%Comparative Analysis of Representation Methods for Classification
%MNIST Dataset
%% Initial Setup
clear; close all; clc;
num_folds = 5;
rng(42); %For reproducibility
save_intermediate = true; 
%% Helper Functions
function images = readMNISTImages(filename)
    fileID = fopen(filename, 'r');
    magic = fread(fileID, 1, 'int32', 0, 'b');
    numImages = fread(fileID, 1, 'int32', 0, 'b');
    numRows = fread(fileID, 1, 'int32', 0, 'b');
    numCols = fread(fileID, 1, 'int32', 0, 'b');
    images = fread(fileID, inf, 'unsigned char');
    images = reshape(images, numCols*numRows, numImages)';
    images = double(images) / 255;
    fclose(fileID);
end

function labels = readMNISTLabels(filename)
    fileID = fopen(filename, 'r');
    magic = fread(fileID, 1, 'int32', 0, 'b');
    numLabels = fread(fileID, 1, 'int32', 0, 'b');
    labels = fread(fileID, inf, 'unsigned char');
    fclose(fileID);
end

function [accuracy, precision, recall, f1, confMat] = evaluateMetrics(trueLabels, predLabels)
    %Calculating accuracy
    selectedDigits = unique(trueLabels);
    accuracy = sum(predLabels == trueLabels) / length(trueLabels);
    
    %Initialising metrics
    nClasses = length(selectedDigits);
    precision = zeros(nClasses, 1);
    recall = zeros(nClasses, 1);
    f1 = zeros(nClasses, 1);
    confMat = zeros(nClasses);
    
    %Calculating confusion matrix and metrics for each class
    for i = 1:nClasses
        class_i_indices = (trueLabels == selectedDigits(i));
        for j = 1:nClasses
            confMat(i, j) = sum(predLabels(class_i_indices) == selectedDigits(j));
        end
        
        tp = confMat(i,i);
        actual_sum = sum(confMat(i, :));
        pred_sum = sum(confMat(:, i));
        
        precision(i) = tp / max(pred_sum, eps);
        recall(i) = tp / max(actual_sum, eps);
        f1(i) = 2 * (precision(i) * recall(i)) / max((precision(i) + recall(i)), eps);
    end
end
%% Load and Prepare Data
%Defining file paths and selected digits
trainImagesFile = 'train-images-idx3-ubyte';
trainLabelsFile = 'train-labels-idx1-ubyte';
testImagesFile = 't10k-images-idx3-ubyte';
testLabelsFile = 't10k-labels-idx1-ubyte';

%Loading data
trainImages = readMNISTImages(trainImagesFile);
trainLabels = readMNISTLabels(trainLabelsFile);
testImages = readMNISTImages(testImagesFile);
testLabels = readMNISTLabels(testLabelsFile);

%Splitting into training and validation sets
cv = cvpartition(size(trainImages, 1), 'HoldOut', 0.2);
X_train = trainImages(cv.training, :);
y_train = trainLabels(cv.training);
X_val = trainImages(cv.test, :);
y_val = trainLabels(cv.test);
X_test = testImages;
y_test = testLabels;

%Displaying dataset info
fprintf('Training set: %d images\n', size(X_train, 1));
fprintf('Validation set: %d images\n', size(X_val, 1));
fprintf('Test set: %d images\n', size(X_test, 1));

%Visualising sample digits
figure;
for i = 0:9
    digitIdx = find(trainLabels == i, 1);
    subplot(2, 5, i+1); % 2x5 grid
    imshow(reshape(trainImages(digitIdx, :), [28, 28])');
    title(sprintf('Digit: %d', i));
end
sgtitle('MNIST Digit Samples');
%% Method 1: Raw Data with kNN
fprintf('\nMETHOD 1: RAW DATA WITH kNN \n');
method_start_time = tic;

%Parameter search
k_values = [1, 3, 5, 7, 9];
best_k = 1;
best_cv_acc = 0;

fprintf('Cross-validation for Raw Data\n');
for k = k_values
    %Initialising cross-validation
    cv = cvpartition(size(X_train, 1), 'KFold', num_folds);
    fold_acc = zeros(num_folds, 1);
    
    %k-fold cross-validation
    for fold = 1:num_folds
        %Getting indices for this fold
        train_idx = cv.training(fold);
        val_idx = cv.test(fold);
        
        %Training model on this fold
        fold_model = fitcknn(X_train(train_idx, :), y_train(train_idx), 'NumNeighbors', k, 'Distance', 'euclidean');
        fold_pred = predict(fold_model, X_train(val_idx, :));
        fold_acc(fold) = sum(fold_pred == y_train(val_idx)) / length(y_train(val_idx));
    end
    
    %Average accuracy across folds
    mean_acc = mean(fold_acc);
    fprintf('k=%d: CV Accuracy=%.4f\n', k, mean_acc);
    
    if mean_acc > best_cv_acc
        best_cv_acc = mean_acc;
        best_k = k;
    end
end

%Final evaluation
final_model = fitcknn(trainImages, trainLabels, 'NumNeighbors', best_k, 'Distance', 'euclidean');
test_pred = predict(final_model, X_test);
[test_acc, test_prec, test_rec, test_f1, test_confMat] = evaluateMetrics(y_test, test_pred);

% Store results
raw_results = struct();
raw_results.method = 'Raw Data';
raw_results.test_acc = test_acc;
raw_results.test_prec = test_prec;  
raw_results.test_rec = test_rec;   
raw_results.test_f1 = test_f1;     
raw_results.confMat = test_confMat;
raw_results.best_k = best_k;
raw_results.num_components = size(X_train, 2);
raw_results.params = 'N/A';

% Display metrics
fprintf('\nRaw Data Results\n');
fprintf('Test Accuracy: %.4f\n', test_acc);
fprintf('Average Precision: %.4f\n', mean(test_prec));
fprintf('Average Recall: %.4f\n', mean(test_rec));
fprintf('Average F1 Score: %.4f\n', mean(test_f1));

%Plotting confusion matrix
figure;
confusionchart(test_confMat, 0:9);
title('Confusion Matrix - Raw Data');

raw_results.time = toc(method_start_time);
fprintf('\nRaw Data computation time: %.2f seconds (%.2f minutes)\n', raw_results.time, raw_results.time/60);
raw_results.samples = size(X_train, 1);

%Save results
if save_intermediate
    save('results_raw.mat', 'raw_results');
end

%Clearing large variables
clear final_model test_pred;

%Forcing garbage collection
java.lang.System.gc();
pause(2);

%% Method 2: PCA with kNN 
fprintf('\nMETHOD 2: PCA WITH kNN\n');
method_start_time = tic;

%Dimensionality Reduction
[X_train_pca, mapping] = compute_mapping(X_train, 'PCA', 0.80);
num_components = size(X_train_pca, 2);
fprintf('Embedded into %d dimensions for 80%% variance\n', num_components);

%Projecting All Datasets
X_val_pca = out_of_sample(X_val, mapping);
X_test_pca = out_of_sample(X_test, mapping);

%kNN Parameter Tuning
best_k = 1;
best_cv_acc = 0;

fprintf('Cross-validation for PCA\n');
for k = [1,3,5,7,9]
    %Initialising cross-validation
    cv = cvpartition(size(X_train_pca, 1), 'KFold', num_folds);
    fold_acc = zeros(num_folds, 1);
    
    %k-fold cross-validation
    for fold = 1:num_folds
        % Get indices for this fold
        train_idx = cv.training(fold);
        val_idx = cv.test(fold);
        
        %Training model on this fold
        fold_model = fitcknn(X_train_pca(train_idx, :), y_train(train_idx), 'NumNeighbors', k, 'Distance', 'cosine');
        fold_pred = predict(fold_model, X_train_pca(val_idx, :));
        fold_acc(fold) = sum(fold_pred == y_train(val_idx)) / length(y_train(val_idx));
    end
    
    %Average accuracy across folds
    mean_acc = mean(fold_acc);
    fprintf('k=%d: CV Accuracy=%.4f\n', k, mean_acc);
    
    if mean_acc > best_cv_acc
        best_cv_acc = mean_acc;
        best_k = k;
    end
end

%Final Evaluation
%Training on full training set (train + val)
X_full_pca = [X_train_pca; X_val_pca];
y_full = [y_train; y_val];

final_model = fitcknn(X_full_pca, y_full, 'NumNeighbors', best_k, 'Distance', 'cosine');
    
%Test evaluation
test_pred = predict(final_model, X_test_pca);
[test_acc, test_prec, test_rec, test_f1, test_confMat] = evaluateMetrics(y_test, test_pred);

%2D Projection
figure('Position', [100, 100, 700, 500]);
gscatter(X_train_pca(:,1), X_train_pca(:,2), y_train, lines(10), '.', 8);
grid on;
title('PCA 2D Projection');
xlabel('PC1'); ylabel('PC2');
legend('Location', 'eastoutside');
axis tight;

%Confusion matrix
figure;
confusionchart(test_confMat, 0:9);
title('Confusion Matrix - PCA');

%Performance Metrics
fprintf('\nPCA Results \n');
fprintf('Test Accuracy: %.4f\n', test_acc);
fprintf('Average Precision: %.4f\n', mean(test_prec));
fprintf('Average Recall: %.4f\n', mean(test_rec));
fprintf('Average F1 Score: %.4f\n', mean(test_f1));

%Storing PCA results
pca_results = struct();
pca_results.method = 'PCA';
pca_results.test_acc = test_acc;
pca_results.test_prec = test_prec;
pca_results.test_rec = test_rec;
pca_results.test_f1 = test_f1;
pca_results.confMat = test_confMat;
pca_results.best_k = best_k;
pca_results.num_components = num_components;
pca_results.params = sprintf('k=%d, n_comp=%d', best_k, num_components);

pca_results.time = toc(method_start_time);
fprintf('\nPCA computation time: %.2f seconds (%.2f minutes)\n', pca_results.time, pca_results.time/60);
pca_results.samples = size(X_train, 1);

%Saving results
if save_intermediate
    save('results_pca.mat', 'pca_results');
end

%Clearing large variables
clear X_train_pca X_val_pca X_test_pca X_full_pca final_model test_pred mapping;
java.lang.System.gc();
pause(2);

%% METHOD 3: KPCA via Nyström with kNN
fprintf('\nMETHOD 3: KPCA via Nyström WITH kNN\n');
method_start_time = tic;

%Parameters
num_landmarks    = 20000;           % m
sigma_value      = 15;              % RBF width
num_components   = 100;             % k
k_candidates     = [1,3,5,7,9];

%Stratified landmark selection
landmark_idx = [];
for digit = 0:9
    idx_digit = find(y_train == digit);
    landmark_idx = [landmark_idx; randsample(idx_digit, num_landmarks/10)];
end
landmarks = X_train(landmark_idx, :);    % [m × d]
  
%Kernel Matrix
D2_W   = pdist2(landmarks, landmarks, 'euclidean').^2;
W      = exp(-D2_W / (2*sigma_value^2));

%C_train: train–landmark [N×m]
D2_Ct  = pdist2(X_train, landmarks, 'euclidean').^2;
C_train = exp(-D2_Ct / (2*sigma_value^2));

%C_val: val–landmark     [N_val×m]
D2_Cv  = pdist2(X_val, landmarks, 'euclidean').^2;
C_val   = exp(-D2_Cv / (2*sigma_value^2));

%C_test: test–landmark   [N_test×m]
D2_Cte = pdist2(X_test, landmarks, 'euclidean').^2;
C_test  = exp(-D2_Cte / (2*sigma_value^2));

%Eigen decomposition
opts.tol   = 1e-4;
opts.maxit = 300;
opts.disp  = 0;
fprintf(' computing top %d eigenpairs via eigs', num_components); tic;
[U_landmarks, S_landmarks] = eigs(W, num_components, 'largestabs', opts);
eig_time = toc;
fprintf(' done in %.1f s\n', eig_time);

lambda   = diag(S_landmarks);                
Lambda_inv_sqrt = diag(1 ./ sqrt(lambda));   

%Forming Nyström embeddings
fprintf(' forming train embedding (%d×%d)', size(X_train,1), num_components); tic;
X_train_emb = C_train * U_landmarks * Lambda_inv_sqrt;  % [N×k]
train_emb_time = toc;
fprintf(' %.1f s\n', train_emb_time);

fprintf(' forming val embedding (%d×%d)', size(X_val,1), num_components); tic;
X_val_emb   = C_val   * U_landmarks * Lambda_inv_sqrt;  % [N_val×k]
val_emb_time = toc;
fprintf(' %.1f s\n', val_emb_time);

fprintf(' forming test embedding (%d×%d)', size(X_test,1), num_components); tic;
X_test_emb  = C_test  * U_landmarks * Lambda_inv_sqrt;  % [N_test×k]
test_emb_time = toc;
fprintf(' %.1f s\n', test_emb_time);

%kNN hyperparameter search with cross-validation
best_k = 1;
best_cv_acc = 0;

fprintf('Cross-validation for KPCA\n');
for k = k_candidates
    %Initialising cross-validation
    cv = cvpartition(size(X_train_emb, 1), 'KFold', num_folds);
    fold_acc = zeros(num_folds, 1);
    
    %k-fold cross-validation
    for fold = 1:num_folds
        % Get indices for this fold
        train_idx = cv.training(fold);
        val_idx = cv.test(fold);
        
        %Training model on this fold
        mdl = fitcknn(X_train_emb(train_idx, :), y_train(train_idx), 'NumNeighbors', k, 'Distance', 'cosine');
        preds = predict(mdl, X_train_emb(val_idx, :));
        fold_acc(fold) = mean(preds == y_train(val_idx));
    end
    
    %Average accuracy across folds
    mean_acc = mean(fold_acc);
    fprintf(' k=%d → CV Acc = %.4f\n', k, mean_acc);
    
    if mean_acc > best_cv_acc
        best_cv_acc = mean_acc;
        best_k = k;
    end
end

%Final training on train+val & test evaluation
X_full_emb = [X_train_emb; X_val_emb];
y_full     = [y_train;      y_val];
final_mdl  = fitcknn(X_full_emb, y_full, 'NumNeighbors', best_k, 'Distance','cosine');
test_preds = predict(final_mdl, X_test_emb);
[test_acc, test_prec, test_rec, test_f1, test_conf] = evaluateMetrics(y_test, test_preds);

fprintf('\nNyström KPCA kNN Results\n');
fprintf(' Test Accuracy:   %.4f\n', test_acc);
fprintf(' Avg Precision:   %.4f\n', mean(test_prec));
fprintf(' Avg Recall:      %.4f\n', mean(test_rec));
fprintf(' Avg F1 Score:    %.4f\n', mean(test_f1));

%Projection plot
figure('Position', [100, 100, 700, 500]);
gscatter(X_train_emb(:,1), X_train_emb(:,2), y_train, lines(10), '.', 8);
grid on;
title('Nyström KPCA 2D Projection');
xlabel('Component 1'); ylabel('Component 2');
legend('Location', 'eastoutside');
axis tight;

%Confusion MAtrix
figure; 
confusionchart(test_conf, 0:9); 
title('Confusion Matrix – Nyström KPCA');

%Storing the results
kpca_results = struct();
kpca_results.method = 'Kernel PCA';
kpca_results.test_acc = test_acc;
kpca_results.test_prec = test_prec;
kpca_results.test_rec = test_rec; 
kpca_results.test_f1 = test_f1;
kpca_results.confMat = test_confMat;
kpca_results.best_k = best_k;
kpca_results.num_components = num_components;
kpca_results.params = sprintf('k=%d, σ=%.1f', best_k, sigma_value);

kpca_results.time = toc(method_start_time);
fprintf('\nKPCA computation time: %.2f seconds (%.2f minutes)\n', kpca_results.time, kpca_results.time/60);
kpca_results.samples = size(X_train, 1);

%Saving results
if save_intermediate
    save('results_kpca.mat', 'kpca_results');
end

%Clearing large variables
clear X_train_emb X_val_emb X_test_emb X_full_emb;
clear D2_W W D2_Ct C_train D2_Cv C_val D2_Cte C_test;
clear U_landmarks S_landmarks lambda Lambda_inv_sqrt landmarks;
java.lang.System.gc();
pause(2);

%% METHOD 4: LLE WITH kNN 
fprintf('\nMETHOD 4: LLE with kNN\n');
method_start_time = tic;

%Parameters
subset_size = 30000;   
embed_dim   = 20;       
num_neigh   = 12;       
reg_param   = 1e-3;     
k_values    = [1,3,5,7,9];
chunk_size  = 2000;     

%Stratified subsampling for LLE mapping
subset_idx = [];
for d = 0:9
    idx = find(y_train==d);
    subset_idx = [subset_idx; randsample(idx, subset_size/10)];
end
X_sub = X_train(subset_idx, :);

%Learning LLE mapping on the subset
[~, lle_map] = compute_mapping(X_sub, 'LLE', embed_dim, num_neigh, reg_param);

%Chunked out-of-sample projection
Ntrain = size(X_train,1);
X_train_lle = zeros(Ntrain, embed_dim);
for i = 1:ceil(Ntrain/chunk_size)
    i1 = (i-1)*chunk_size + 1;
    i2 = min(i*chunk_size, Ntrain);
    X_train_lle(i1:i2, :) = out_of_sample(X_train(i1:i2, :), lle_map);
end

Nval = size(X_val,1);
X_val_lle = zeros(Nval, embed_dim);
for i = 1:ceil(Nval/chunk_size)
    i1 = (i-1)*chunk_size + 1;
    i2 = min(i*chunk_size, Nval);
    X_val_lle(i1:i2, :) = out_of_sample(X_val(i1:i2, :), lle_map);
end

Ntest = size(X_test,1);
X_test_lle = zeros(Ntest, embed_dim);
for i = 1:ceil(Ntest/chunk_size)
    i1 = (i-1)*chunk_size + 1;
    i2 = min(i*chunk_size, Ntest);
    X_test_lle(i1:i2, :) = out_of_sample(X_test(i1:i2, :), lle_map);
end

%kNN on the 20-D LLE features
best_k = 1;
best_cv_acc = 0;

fprintf('Cross-validation for LLE\n');
for k = k_values
    %Initialising cross-validation
    cv = cvpartition(size(X_train_lle, 1), 'KFold', num_folds);
    fold_acc = zeros(num_folds, 1);
    
    %k-fold cross-validation
    for fold = 1:num_folds
        %Getting indices for this fold
        train_idx = cv.training(fold);
        val_idx = cv.test(fold);
        
        %Training model on this fold
        mdl = fitcknn(X_train_lle(train_idx, :), y_train(train_idx), 'NumNeighbors', k, 'Distance', 'cosine');
        preds = predict(mdl, X_train_lle(val_idx, :));
        fold_acc(fold) = mean(preds == y_train(val_idx));
    end
    
    %Average accuracy across folds
    mean_acc = mean(fold_acc);
    fprintf(' k=%d → CV Acc = %.4f\n', k, mean_acc);
    
    if mean_acc > best_cv_acc
        best_cv_acc = mean_acc;
        best_k = k;
    end
end

%Final evaluation on train+val and test
fullX   = [X_train_lle; X_val_lle];
fully   = [y_train;      y_val];
final_mdl = fitcknn(fullX, fully, 'NumNeighbors', best_k, 'Distance', 'cosine');
test_pred = predict(final_mdl, X_test_lle);

[test_acc, test_prec, test_rec, test_f1, test_confMat] = evaluateMetrics(y_test, test_pred);

%Projection plot
figure('Position', [100, 100, 700, 500]);
gscatter(X_train_lle(:,1), X_train_lle(:,2), y_train, lines(10), '.', 8);
grid on;
title('LLE 2D Projection');
xlabel('Component 1'); ylabel('Component 2');
legend('Location', 'eastoutside');
axis tight;

%Plot confusion matrix
figure;
confusionchart(test_confMat, 0:9);
title('Confusion Matrix – LLE');

%Display metrics
fprintf('\nLLE Results\n');
fprintf('Test Accuracy: %.4f\n', test_acc);
fprintf('Average Precision: %.4f\n', mean(test_prec));
fprintf('Average Recall: %.4f\n', mean(test_rec));
fprintf('Average F1 Score: %.4f\n', mean(test_f1));

%Storing the results
lle_results = struct();
lle_results.method         = 'LLE';
lle_results.test_acc       = test_acc;
lle_results.test_prec      = test_prec;
lle_results.test_rec       = test_rec;
lle_results.test_f1        = test_f1;
lle_results.confMat        = test_confMat;
lle_results.best_k         = best_k;
lle_results.num_components = embed_dim;
lle_results.params         = sprintf('neighbors=%d, reg=%.1e', num_neigh, reg_param);

lle_results.time = toc(method_start_time);
fprintf('\nLLE computation time: %.2f seconds (%.2f minutes)\n', lle_results.time, lle_results.time/60);
lle_results.samples = subset_size;

%Saving results
if save_intermediate
    save('results_lle.mat', 'lle_results');
end
%% Comparative Analysis

%Preparing data
methods = {'Raw Data', 'PCA', 'Kernel PCA', 'LLE'};
metrics = struct();

%Calculating averages
metrics.accuracy = [raw_results.test_acc, pca_results.test_acc, kpca_results.test_acc, lle_results.test_acc];
metrics.precision = [mean(raw_results.test_prec), mean(pca_results.test_prec), mean(kpca_results.test_prec), mean(lle_results.test_prec)];
metrics.recall = [mean(raw_results.test_rec), mean(pca_results.test_rec), mean(kpca_results.test_rec), mean(lle_results.test_rec)];
metrics.f1 = [mean(raw_results.test_f1), mean(pca_results.test_f1), mean(kpca_results.test_f1), mean(lle_results.test_f1)];

%Metrics Table
fprintf('\n Performance Metrics Table \n');
fprintf('%-12s %-10s %-10s %-10s %-10s\n', 'Method', 'Accuracy', 'Precision', 'Recall', 'F1 Score');
disp('-----------------------------------------------------');
for i = 1:4
    fprintf('%-12s %-10.4f %-10.4f %-10.4f %-10.4f\n', ...
            methods{i}, metrics.accuracy(i), metrics.precision(i), ...
            metrics.recall(i), metrics.f1(i));
end

%Time table
fprintf('\n Computation Time Comparison\n');
fprintf('%-12s %-15s %-15s\n', 'Method', 'Time (seconds)', 'Time (minutes)');
disp('-------------------------------------------------');
fprintf('%-12s %-15.2f %-15.2f\n', 'Raw Data', raw_results.time, raw_results.time/60);
fprintf('%-12s %-15.2f %-15.2f\n', 'PCA', pca_results.time, pca_results.time/60);
fprintf('%-12s %-15.2f %-15.2f\n', 'Kernel PCA', kpca_results.time, kpca_results.time/60);
fprintf('%-12s %-15.2f %-15.2f\n', 'LLE', lle_results.time, lle_results.time/60);

%Individual Metric Plots

%Creating a plotting function
function createMetricPlot(data, metricName, methods, ylimits)
    figure('Position', [100, 100, 600, 400]);
    b = bar(data);
    set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 45);
    ylabel('Score');
    title([metricName ' Comparison']);
    ylim(ylimits);
    grid on;

    xtips = b.XEndPoints;
    ytips = b.YEndPoints;
    labels = arrayfun(@(x) sprintf('%.4f', x), data, 'UniformOutput', false);
    text(xtips, ytips, labels, 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', 'FontSize', 10);
end

%Generating plots for all metrics
createMetricPlot(metrics.accuracy, 'Accuracy', methods, [0.8 1.0]);
createMetricPlot(metrics.precision, 'Precision', methods, [0.8 1.0]);
createMetricPlot(metrics.recall, 'Recall', methods, [0.8 1.0]);
createMetricPlot(metrics.f1, 'F1 Score', methods, [0.8 1.0]);

%Class-wise F1 Scores Plot
figure('Position', [100, 100, 1000, 500]);
class_f1 = [raw_results.test_f1, pca_results.test_f1, kpca_results.test_f1, lle_results.test_f1];

b = bar(class_f1, 'grouped');
set(gca, 'XTickLabel', 0:9);
xlabel('Digit');
ylabel('F1 Score');
title('Class-wise F1 Scores');
legend({'Raw Data', 'PCA', 'Kernel PCA', 'LLE'}, 'Location', 'northwest');
ylim([0.7, 1.08]);
grid on;

% Annotate each bar with vertical-rotated label
for j = 1:numel(b)
    x = b(j).XEndPoints;
    y = b(j).YEndPoints;
    for i = 1:length(x)
        val = class_f1(i, j);
        text(x(i), y(i) + 0.01, sprintf('%.3f', val), ...
             'Rotation', 90, ...
             'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'bottom', ...
             'FontSize', 8);
    end
end

