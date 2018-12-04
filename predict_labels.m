function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)
n = size(train_inputs,1); 
m = size(test_inputs, 1); 
p = size(train_inputs,2);

% DIMENSIONALITY REDUCTION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% step 1 
% feature selection based on NaN topics and topic frequencies 
% topics_kept = csvread('topics_kept_features.csv'); 
demo_inputs_train = train_inputs(:, 1:21); 
demo_inputs_test = test_inputs(:, 1:21); 

topics_inputs_train = train_inputs(:, 22:p); 
topics_inputs_test = test_inputs(:, 22:p); 


% PCA on topics_inputs_train
% 1. not sparse - mc for PCA is ok! 
all_topics = vertcat(topics_inputs_train, topics_inputs_test); 

[~,score,latent,~,~,~] = pca(all_topics); 
summ = sum(latent); 
sum_eigen_value = 0; 
variance_explained = zeros(size(latent,1),1); 

for i = 1:size(latent,1) 
    sum_eigen_value = sum_eigen_value + latent(i); 
    percent_var_explained = 100*sum_eigen_value/summ; 
    variance_explained(i) = percent_var_explained; 
end 

pc_num = sum(variance_explained < 99.5); % number of PC used for all_topics 

train_topics = score(1:n,1:pc_num); 
test_topics = score((n+1):(n+m), 1:pc_num);

% reduced train_inputs and test_inputs
train_inputs = horzcat(demo_inputs_train, train_topics); 
test_inputs = horzcat(demo_inputs_test, test_topics);     


% KNN Regression 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. rescale train_inputs 
% max_min rescale

%%% below commented out
%{
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
%}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. find distance 
% Euclidean distance matrix
% e.g. row 1: m = 1 instance's distance to all n training data 
%{
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
   
    KNN_output_demo_test(row,:) = sum(train_labels(I_demo_test, :) ...
        .* weights_demo_test', 1) / sum_weights_demo_test; 
    KNN_output_topics_test(row,:) = sum(train_labels(I_topics_test, :) ...
        .* weights_topics_test', 1) / sum_weights_topics_test; 

end

pred_labels_KNN = (KNN_output_demo_test + KNN_output_topics_test) /2;

% error_KNN = error_metric(pred_labels_KNN, test_labels);

%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Ridge regression 
% can tune: alpha, lambda 
% pred_labels = zeros(m, 9); 

pred_labels = zeros(m, 9); 
for i = 1:9 
    [B, Stats] = lasso(train_inputs, train_labels(:,i), 'cv', 5, 'Alpha', 1); 
    % disp(B(:,1:5))
    % disp(Stats)
    % lassoPlot(B, Stats, 'PlotType', 'CV')
    ds.Lasso = B(:,Stats.IndexMinMSE);
    ds.Intercept = Stats.Intercept(Stats.IndexMinMSE); 
   
    
    % disp(ds)
    pred_labels(:,i) = test_inputs * ds.Lasso + ds.Intercept; 
    
    %{
    Betas = B(:,Stats.IndexMinMSE) > 0;
    Coeff_Num = sum(B(:,Stats.IndexMinMSE) > 0);
    Number_Lasso_Coefficients = mean(Coeff_Num);
    disp(Number_Lasso_Coefficients)

    MSE = Stats.MSE(:, Stats.IndexMinMSE);
    disp(median(MSE))
    %}
end 













% alpha = 0.00001; % make it ridge 
% error_holder = zeros(size(alpha,2),1); 

%{
p = size(train_inputs, 2); 
best_lam_holder = zeros(9,1); 
best_coef_holder = zeros(p, 9); 
best_c_holder = zeros(1,9); 

% train_inputs = vertcat(train_inputs, test_inputs); 
n = size(train_inputs,1); 
c = cvpartition(n,'kfold',5); 

for cv = 1:5 % 5-fold cv 
    cv_test_inputs = train_inputs(c.test(cv),:); 
    cv_train_inputs = train_inputs(c.training(cv),:); 
    
    cv_test_labels = train_labels(c.test(cv),:); 
    cv_train_labels = train_labels(c.training(cv),:); 
    m = size(cv_test_labels,1); 
    
    cv_lam_holder = zeros(9,1); 
    cv_coef_holder = zeros(p, 9); 
    cv_c_holder = zeros(1,9); 
    
for i = 1:9 % loop thru features 
    y = cv_train_labels(:,i); 
    lam = 0.1:0.1:10; % size: 1 * l 
    b = ridge(y,cv_train_inputs,lam, 0);
    % c - constant term 
    constant = b(1,:); % size: 1 * l 
    coef = b(2:size(b,1),:);  % size: p * l 
    % test_inputs - m * p 
    % pred_labels_ridge - m * l 
    pred_labels_ridge = cv_test_inputs * coef + repmat(constant, m, 1); 
    
    this_y_test_labels = zeros(m, 9); 
    this_y_test_labels(:,i) = cv_test_labels(:,i); 
    
    lowest_y_error = error_metric(pred_labels_ridge(:,1), this_y_test_labels); 
    index = 1; 
    for lambda = 2:size(lam, 2) 
        this_y_error = error_metric(pred_labels_ridge(:,lambda), this_y_test_labels); 
        if this_y_error < lowest_y_error
            lowest_y_error = this_y_error; 
            index = lambda; 
        end  
    end 
    best_lam_holder(i) = lam(index); 
    best_coef_holder(:,i) = coef(:,index); 
    best_c_holder(i) = constant(index); 
end

cv_lam_holder = cv_lam_holder + best_lam_holder; 
cv_coef_holder = cv_coef_holder + best_coef_holder; 
cv_c_holder = cv_c_holder + best_c_holder; 
cv
end 

end_lam_holder = cv_lam_holder ./ 5; 
end_coef_holder = cv_coef_holder ./ 5; 
end_c_holder = cv_c_holder ./ 5; 


m = size(test_inputs,1); 
pred_labels_ridge = zeros(m, 9); 
for i = 1:9 % loop thru features 
    pred_labels_ridge(:,i) = test_inputs * end_coef_holder(:,i) + repmat(end_c_holder(i), m, 1); 
end 

error_metric(pred_labels_ridge, test_labels); 
%}

%{
for i = 1:9 % loop thru features 
    y = train_labels(:,i); 
    lam = 1e-6:1e-6:5e-4; 
    [B,FitInfo] = lasso(train_inputs, y ,'Alpha', alpha, 'CV', 5, 'Lambda',lam);
    idxLambda1MSE = FitInfo.Index1SE;
    best_lam_holder(i) = lam(idxLambda1MSE); 
    % coef = B(:,idxLambda1MSE);
    % coef0 = FitInfo.Intercept(idxLambda1MSE); 
    % pred_labels(:,i) = test_inputs * coef + coef0;  
end
%}

% end 
 
% error_metric(pred_labels, test_labels)

% pred_labels = (pred_labels_svm + pred_labels_ridge) ./ 2; 


% 2. RF 
%{
numTree = 500; 
pred_labels_tree = zeros(m, 9); 
bestHyperparameters = csvread('bestHyperparams.csv'); 
    
for i = 1: 9 % loop thru health outcomes 
    y = train_labels(:,i); 

    Mdl = TreeBagger(numTree,train_inputs,y,'Method','regression',...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS);

    pred_labels_tree(:,i) = predict(Mdl, test_inputs); 
end 
%}

% error_tree = error_metric(pred_labels_tree, test_labels);

% 3. RBF NN 
% net = newrb(train_inputs', train_labels'); 
% pred_labels_rbf = (sim(net,test_inputs'))'; 

% error_rbf = error_metric(pred_labels_rbf, test_labels);

end


















