function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)
n = size(train_inputs,1); 
m = size(test_inputs, 1); 
% sample = min(n, m); 

% DIMENSIONALITY REDUCTION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 1 
% all_kept_features = csvread('all_kept_features.csv'); 
topics_kept = csvread('topics_kept_features.csv'); 
demo_kept = csvread('demo_kept_features.csv'); 

topics_inputs_train = train_inputs(:, topics_kept); 
topics_inputs_test = test_inputs(:, topics_kept); 

all_topics = vertcat(topics_inputs_train, topics_inputs_test); 

demo_inputs_train = train_inputs(:, demo_kept); 
demo_inputs_test = test_inputs(:, demo_kept); 


% step 2 - PCA feature number reduction 
% mean center all_topics 
mu_all_topics = zeros(1, size(all_topics,2)); 
for i = 1:size(all_topics,2)
    mu_all_topics(i) = mean(all_topics(:,i)); 
end 

mc_all_topics = all_topics - repmat(mu_all_topics,size(all_topics, 1),1); 

[~,S,~] = svd(mc_all_topics); 
dim_S = min(size(S,1), size(S,2)); 

square_matrix = S(1:dim_S,1:dim_S); 

eigenvals = square_matrix * square_matrix; 
summ = trace(eigenvals); 
sum_eigen_value = 0; 
variance_explained = zeros(dim_S,1); 
for i = 1:dim_S 
    sum_eigen_value = sum_eigen_value + eigenvals(i,i); 
    percent_var_explained = 100*sum_eigen_value/summ; 
    variance_explained(i) = percent_var_explained; 
end 

pc_num = sum(variance_explained < 99.9); % number of PC used for all_topics 

[~,score,~,~,~,~] = pca(all_topics); 

PC_topics = score(:, 1:pc_num); 

train_topics = PC_topics(1:n, :); 
test_topics = PC_topics((n+1):(n+m), :); 

train_inputs = horzcat(demo_inputs_train, train_topics); 
test_inputs = horzcat(demo_inputs_test, test_topics);                                     

% KNN Regression 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. rescale train_inputs 
% max_min rescale

for i = 1:size(train_inputs, 2)
    min_ = min(train_inputs(:,i)); 
    max_ = max(train_inputs(:,i)); 
    train_inputs(:,i) = (train_inputs(:,i) - min_) / (max_-min_);  
end


% 1. rescale test_inputs 
% max_min rescale

for i = 1:size(test_inputs, 2)
    min_ = min(test_inputs(:,i)); 
    max_ = max(test_inputs(:,i)); 
    test_inputs(:,i) = (test_inputs(:,i) - min_) / (max_-min_);  
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. calc KNN output 
% 2 feature blocks: demo & topics 
K = 50; 
P = 2; 
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

end















