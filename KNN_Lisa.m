cd '/Users/lisazhao/Desktop/CIS 520/project_kit'; 
train = csvread('training_data.csv', 1, 0); 
load('training_data.mat'); 

demo_ses_features_ = train_inputs(:,1:21); 
topic_features = train_inputs(:,22:2021); 
n = size(train,1); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% demo_ses_features not full rank - 1 col is linearly dependent 
[~,colind] = rref(demo_ses_features_);
% demo_ses_features = demo_ses_features_(:, colind); 

nan_topics = zeros(1, 2000); 
for row = 1:2000 
    nan_topics(row) = max(ismissing(topics(row,:))); 
end 

topics_with_nan = find(nan_topics); % 458 

% keep the above 15% topics
sums = sum(topic_features,1); 
freq_topics = sums>0.15;

% remove the with nan topics 
freq_topics(:,topics_with_nan) = 0; 

topics_keep = find(freq_topics == 1); 

topics_keep = topics_keep + 21; 
all_kept_features = [colind topics_keep]; 

cd '/Users/lisazhao/Desktop/CIS 520/project_kit'; 
csvwrite('all_kept_features.csv',all_kept_features); 

train_inputs = train_inputs(:, all_kept_features); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CV 

% returns the indices cvIndices after applying cvMethod on N observations 
% using M as the selection parameter.
% M = 20; 
% cvIndices = crossvalind('Kfold',n,M); 

% cv_train_inputs = resc_redc_X(cvIndices ~= 1, :); 
% cv_train_labels = train_labels(cvIndices ~= 1, :); 
% test_inputs = resc_redc_X(cvIndices == 1, :); 
% true_labels = train_labels(cvIndices == 1, :); 

% KNN Regression 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. rescale train_inputs 
% max_min rescale

for i = 1:size(train_inputs, 2)
    min_ = min(train_inputs(:,i)); 
    max_ = max(train_inputs(:,i)); 
    train_inputs(:,i) = (train_inputs(:,i) - min_) / (max_-min_);  
end

% mean rescale 
% resc_redc_X = reduced_X; 

% for i = 1:size(reduced_X, 2)
%     avg = mean(reduced_X(:,i)); 
%     max_ = max(reduced_X(:,i)); 
%     min_ = min(reduced_X(:,i)); 
    
%     resc_redc_X(:,i) = (reduced_X(:,i) - avg) / (max_-min_);  
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. find distance 
% Euclidean distance matrix
% e.g. row 1: n = 1 instance's distance to all n training data 
all_train_distance_demo = zeros(n, n);  
all_train_distance_topics = zeros(n, n);

for i = 1:n % loop thru each row 
   for j = 1:n % loop thru each col 
    all_train_distance_demo(i, j) = sqrt(sum((train_inputs(i,1:20) - train_inputs(j,1:20)) .^2)); 
    
    all_train_distance_topics(i, j) = ...
        sqrt(sum((train_inputs(i,21:size(train_inputs,2)) - train_inputs(j,21:size(train_inputs,2))) .^2)); 
   end                      
end 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. calc KNN output 
% 2 feature blocks: demo & topics 

% CV 
P = 2; 
F = 10; 
indices = crossvalind('Kfold',n,F);

KNN = [10, 20, 30, 40, 50,100,150,200,250,300,350,400,450]; 
cv_error_holder = zeros(1, size(KNN,2)); 

for param = 1:size(KNN,2) % loop thru values of K 
    k = KNN(param); % the param used in this loop 
    
    sum_cv_error = 0; 
for fold = 1:F % loop thru cv folds 
cv_train_train = train_inputs(indices ~= F, :); 
cv_test_train = train_inputs(indices == F, :); 

cv_train_labels = train_labels(indices ~= F, :); 
cv_test_labels = train_labels(indices == F, :); 

m = size(cv_test_train, 1); 
n = size(cv_train_train, 1); 

% Euclidean distance matrix
% e.g. row 1: m = 1 instance's distance to all n training data 
all_test_distance_demo = zeros(m, n);  
all_test_distance_topics = zeros(m, n);

num_fet = size(cv_test_train, 2); 
for i = 1:m % loop thru each row 
   for j = 1:n % loop thru each col 
    all_test_distance_demo(i, j) = sqrt(sum((cv_test_train(i,1:20) - cv_train_train(j,1:20)) .^2)); 
    
    all_test_distance_topics(i, j) = sqrt(sum((cv_test_train(i,21:num_fet) - cv_train_train(j,21:num_fet)) .^2)); 
   end                      
end 


KNN_output_demo_test = zeros(m,9); 
KNN_output_topics_test = zeros(m, 9); 

for row = 1:m
    [B_demo_test, I_demo_test] = mink(all_test_distance_demo(row,:), (k+1)); 
    [B_topics_test, I_topics_test] = mink(all_test_distance_topics(row,:), (k+1)); 
    
    % to exclude itself 
    B_demo_test = B_demo_test(2:(k+1)); 
    I_demo_test = I_demo_test(2:(k+1)); 
    
    B_topics_test = B_topics_test(2:(k+1)); 
    I_topics_test = I_topics_test(2:(k+1)); 
    
    weights_demo_test = 1 ./ (B_demo_test.^P); % w = 1/d^p
    sum_weights_demo_test = sum(weights_demo_test); % for normalization 
    
    weights_topics_test = 1 ./ (B_topics_test.^P); % w = 1/d^p
    sum_weights_topics_test = sum(weights_topics_test); % for normalization 
   
    KNN_output_demo_test(row,:) = sum(cv_train_labels(I_demo_test, :) .* weights_demo_test', 1) / sum_weights_demo_test; 
    KNN_output_topics_test(row,:) = sum(cv_train_labels(I_topics_test, :) .* weights_topics_test', 1) / sum_weights_topics_test; 
end 
  
cv_pred_labels_train = (KNN_output_demo_test + KNN_output_topics_test) /2; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sum_cv_error = sum_cv_error + error_metric(cv_pred_labels_train, cv_test_labels); 
end 
cv_error_holder(param) = sum_cv_error / F; 
param
end 





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% No CV 
KNN_output_demo = zeros(n,9); 
KNN_output_topics = zeros(n, 9); 

for row = 1:n
    [B_demo, I_demo] = mink(all_train_distance_demo(row,:), (K+1)); 
    [B_topics, I_topics] = mink(all_train_distance_topics(row,:), (K+1)); 
    
    % to exclude itself 
    B_demo = B_demo(2:(K+1)); 
    I_demo = I_demo(2:(K+1)); 
    
    B_topics = B_topics(2:(K+1)); 
    I_topics = I_topics(2:(K+1)); 
    
    weights_demo = 1 ./ (B_demo.^P); % w = 1/d^p
    sum_weights_demo = sum(weights_demo); % for normalization 
    
    weights_topics = 1 ./ (B_topics.^P); % w = 1/d^p
    sum_weights_topics = sum(weights_topics); % for normalization 
   
    KNN_output_demo(row,:) = sum(train_labels(I_demo, :) .* weights_demo', 1) / sum_weights_demo; 
    KNN_output_topics(row,:) = sum(train_labels(I_topics, :) .* weights_topics', 1) / sum_weights_topics; 
end 
  
pred_labels_train = (KNN_output_demo + KNN_output_topics) /2; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_error = error_metric(pred_labels_train, train_labels); 







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for test_inputs
test_inputs = test_inputs(:, all_kept_features); 

% 1. rescale test_inputs 
% max_min rescale

for i = 1:size(test_inputs, 2)
    min_ = min(test_inputs(:,i)); 
    max_ = max(test_inputs(:,i)); 
    test_inputs(:,i) = (test_inputs(:,i) - min_) / (max_-min_);  
end

m = size(test_inputs, 1); 

% 2. find distance 
% Euclidean distance matrix
% e.g. row 1: m = 1 instance's distance to all n training data 
all_test_distance_demo = zeros(m, n);  
all_test_distance_topics = zeros(m, n);

num_fet = size(test_inputs, 2); 
for i = 1:m % loop thru each row 
   for j = 1:n % loop thru each col 
    all_test_distance_demo(i, j) = sqrt(sum((test_inputs(i,1:20) - train_inputs(j,1:20)) .^2)); 
    
    all_test_distance_topics(i, j) = sqrt(sum((test_inputs(i,21:num_fet) - train_inputs(j,21:num_fet)) .^2)); 
   end                      
end 


% 3. calc KNN output 
% 2 feature blocks: demo & topics 
KNN_output_demo_test = zeros(m,9); 
KNN_output_topics_test = zeros(m, 9); 

for row = 1:m
    [B_demo_test, I_demo_test] = mink(all_test_distance_demo(row,:), (K+1)); 
    [B_topics_test, I_topics_test] = mink(all_test_distance_topics(row,:), (K+1)); 
    
    % to exclude itself 
    B_demo_test = B_demo_test(2:(K+1)); 
    I_demo_test = I_demo_test(2:(K+1)); 
    
    B_topics_test = B_topics_test(2:(K+1)); 
    I_topics_test = I_topics_test(2:(K+1)); 
    
    weights_demo_test = 1 ./ (B_demo_test.^P); % w = 1/d^p
    sum_weights_demo_test = sum(weights_demo_test); % for normalization 
    
    weights_topics_test = 1 ./ (B_topics_test.^P); % w = 1/d^p
    sum_weights_topics_test = sum(weights_topics_test); % for normalization 
   
    KNN_output_demo_test(row,:) = sum(train_labels(I_demo_test, :) .* weights_demo_test', 1) / sum_weights_demo_test; 
    KNN_output_topics_test(row,:) = sum(train_labels(I_topics_test, :) .* weights_topics_test', 1) / sum_weights_topics_test; 
end 
  
pred_labels = (KNN_output_demo_test + KNN_output_topics_test) /2; 
























