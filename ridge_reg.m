% Ridge regularization 
% input: reduced train_inputs 
n = size(train_inputs, 1); 
p = size(train_inputs, 2); 
m = size(test_inputs, 1); 


% The value Alpha = 1 represents lasso regression, Alpha close to 0 
% approaches ridge regression, and other values represent elastic net 
% optimization

% can tune 
% 1. Alpha 
% 2. Lambda 

% all Train 
pred_all_Train_holder = zeros(size(train_labels,1),9); 
coef_holder = zeros(size(train_inputs, 2), 9); 
coef0_holder = zeros(1, 9); 

for i = 1:9 
    y = train_labels(:,i); 
    lam = 0.1:0.1:1; 
    [B,FitInfo] = lasso(train_inputs, y ,'Alpha', 0.01, 'CV', 10, 'Lambda',lam);
    idxLambdaMinMSE = FitInfo.IndexMinMSE;
    coef_holder(:,i) = B(:,idxLambdaMinMSE);
    coef0_holder(i) = FitInfo.Intercept(idxLambdaMinMSE);
    
    pred_all_Train_holder(:,i) = train_inputs * coef_holder(:,i) + coef0_holder(i);
end

train_error = error_metric(pred_all_Train_holder, train_labels); 

cd '/Users/lisazhao/Desktop/CIS 520/project_kit'; 
csvwrite('coef_holder.csv',coef_holder);
csvwrite('coef0_holder.csv',coef0_holder);


% hold out 
c = cvpartition(n,'HoldOut',0.1);

idxTrain = training(c,1);
idxTest = ~idxTrain;
XTrain = train_inputs(idxTrain,:);
yTrain = train_labels(idxTrain,:);

all_XTest = train_inputs(idxTest,:);
all_yTest = train_labels(idxTest,:);

pred_yTest_holder = zeros(size(all_XTest,1),9); 
ho_coef_holder = zeros(size(XTrain, 2), 9); 
ho_coef0_holder = zeros(1, 9); 

for i = 1:9 
    y = yTrain(:,i); 
    lam = 0.1:0.1:1; 
    [B,FitInfo] = lasso(XTrain, y ,'Alpha', 0.01, 'CV', 10, 'Lambda',lam);
    idxLambdaMinMSE = FitInfo.IndexMinMSE;
    ho_coef_holder(:,i) = B(:,idxLambdaMinMSE);
    ho_coef0_holder(i) = FitInfo.Intercept(idxLambdaMinMSE);
    
    pred_yTest_holder(:,i) = all_XTest * ho_coef_holder(:,i) + ho_coef0_holder(i);
end

holdout_error = error_metric(pred_yTest_holder,all_yTest);  

csvwrite('ho_coef_holder.csv',ho_coef_holder);
csvwrite('ho_coef0_holder.csv',ho_coef0_holder);



% to tune alpha 
pred_labels = zeros(m, 9); 
alpha = 0.1:0.1:1;
coef_holder(1:size(alpha,2)) = {zeros(p, 9)};
coef0_holder = zeros(size(alpha,2), 1, 9); 
err_holder = zeros(size(alpha,2), 1);

for j = 1:size(alpha,2) % loop thru alpha
for i = 1:9 % loop thru features 
    y = train_labels(:,i); 
    lam = 0.1:0.05:1; 
    [B,FitInfo] = lasso(train_inputs, y ,'Alpha', alpha(i), 'CV', 10, 'Lambda',lam);
    idxLambda1MSE = FitInfo.Index1SE;
    coef_holder{j}(:, i) = B(:,idxLambda1MSE);
    coef0_holder(j, i) = FitInfo.Intercept(idxLambda1MSE);
    
    pred_labels(:,i) = test_inputs * coef_holder{j}(:, i) + coef0_holder(j, i); 
end
err_holder(j) = error_metric(pred_labels, test_labels); 
end





















