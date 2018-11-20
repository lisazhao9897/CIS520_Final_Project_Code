cd '/Users/lisazhao/Desktop/CIS 520/project_kit'; 
train = csvread('training_data.csv', 1, 0); 
load('training_data.mat'); 

demo_features = ["demo_pcblack", "demo_pcfemale", "demo_pchisp", ...
    "demo_pcwht", "demo_under18", "demo_65over", "largemetro", ...
    "mediummetro", "metrononmetro", "micro", "nchs_2013", "ses_edu_coll",...
    "ses_foodenvt", "ses_incomeratio", "ses_log_hhinc", "ses_pcaccess", ...
    "ses_pcexerciss", "ses_pchousing", "ses_pcrural", "ses_pcunemp", "smallmetro"]; 


demo_ses_features = train_inputs(:,1:21); 
topic_features = train_inputs(:,22:2021); 
n = size(train,1); 

% variable importance ranks 
imp_topics = []; 
X = topic_features; 
t = templateTree('MaxNumSplits',1);
for i = 1:9 % loop thru the 9 health outcomes
    Y = train_labels(:,i); 
    ens = fitrensemble(X,Y,'Method','LSBoost','Learners',t);
    imp = predictorImportance(ens); 
    [srt, idxSrt]  = sort(imp);
    imp_topic_cols = idxSrt(imp > 0); 
    imp_topics = [imp_topics, imp_topic_cols]; 
end 
imp_topics = unique(imp_topics); 

% 1019 * 698 
reduced_X = [demo_ses_features,topic_features(:,imp_topics)]; 

% [coeff,score,~,~,~,mu] = pca(reduced_X); 
% Each column of coeff contains coefficients for one principal component. 
% The columns are in the order of descending component variance, latent.

% choose p = n = 1019 - i.e. the first 1019 PCs 
% PC_use = coeff(:,1:(n-size(demo_features,2))); 



% mean-center X 
% mc_X_topics_freq = topics_freq_features - repmat(mu,n,1); 

% [U,S,V] = svd(mc_X_topics_freq); 
% square_matrix = S(:,1:n); 

% eigenvals = square_matrix * square_matrix; 
% sum = trace(eigenvals); 
% sum_eigen_value = 0; 
% variance_explained = zeros(n,1); 
% for i = 1:n 
%     sum_eigen_value = sum_eigen_value + eigenvals(i,i); 
%     percent_var_explained = 100*sum_eigen_value/sum; 
%     variance_explained(i) = percent_var_explained; 
% end 








