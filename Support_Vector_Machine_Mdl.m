% You can set your own kernel function, for example, kernel, by setting 
% 'KernelFunction','kernel'. kernel must have the following form:
% function G = kernel(U,V) https://www.mathworks.com/help/stats/fitrsvm.html
pred_labels_svm = zeros(m, 9); 
for i = 1:9 
    Y = train_labels(:,i); 
    Mdl = fitrsvm(train_inputs,Y,'OptimizeHyperparameters', 'all',...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus')); 
    pred_labels_svm(:,i) = predict(Mdl, test_inputs); 
end 

% 'OptimizeHyperparameters', 'all'
error_svm = error_metric(pred_labels_svm, test_labels); 



Mdl = fitrsvm(train_inputs,Y,'KernelFunction','polynomial',...
        'KernelScale','auto','Standardize',true,  ...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus')); 








