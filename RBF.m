% RBF 
% begin with dim_reduced train_inputs 

p = size(train_inputs, 2); 
n = size(train_inputs, 1); 
neurons = min(400, round(n * 2 / 3)); 

% 10-Fold CV 
F = 10; 
indices = crossvalind('Kfold',n,F);
sum_cv_error = 0; 
for fold = 1:F 
    cv_train_train = train_inputs(indices ~= fold, :); 
    cv_test_train = train_inputs(indices == fold, :); 
    
    cv_train_labels = train_labels(indices ~= fold, :); 
    cv_test_labels = train_labels(indices == fold, :); 
    
    m = size(cv_test_train, 1); 
    nn = size(cv_train_train, 1); 
    
    neurons = min(200, round(nn * 2 / 3)); 
    this_net = newrb(cv_train_train',cv_train_labels', 0, 2, neurons, 25); 
    
    this_net.performFcn = 'crossentropy'; 
    % this_net.trainParam.epochs = neurons; 
    this_net.performParam.regularization = 0.5; 
        
    for l = 1:size(this_net.layers, 1)
        % this_net.layers{l}.transferFcn = 'radbas';
        % net.layers{l}.transferFcn = 'logsig';
        % net.layers{1}.transferFcn = 'tansig';
        net.layers{1}.transferFcn = 'purelin';  
    end
    
    this_pred_train_labels = zeros(m, 9);
    
    for row = 1:m 
    this_input = cv_test_train(row, :)'; 
    this_pred_train_labels(row, :) = (sim(this_net, this_input))'; 
    end
    error_metric(this_pred_train_labels, cv_test_labels)
    % sum_cv_error = sum_cv_error + error_metric(this_pred_train_labels,
    % cv_test_labels); 
    
    
    
    
    
    fold 
end

cv_error = sum_cv_error / F; 






















