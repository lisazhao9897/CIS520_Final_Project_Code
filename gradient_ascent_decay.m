function [weights,error_per_iter] = gradient_ascent_decay(Xtrain,Ytrain,initial_step_size,iterations)

    % Function to perform gradient ascent with a decaying step size for
    % logistic regression.
    % Usage: [weights,error_per_iter] = gradient_ascent(Xtrain,Ytrain,step_size,iterations)
    
    % The parameters to this function are exactly the same as the
    % parameters to gradient ascent with fixed step size.
    
    % initial_step_size : This parameter refers to the initial value of the step
    % size. The actual step size to update the weights will be a value
    % that is (initial_step_size * some function that decays over time)
    % some good choices for this function might by 1/n or 1/sqrt(n).
    % Experiment with such functions, and initial step size until you get
    % good performance.
    
    weights = ones(size(Xtrain,2),1); % P x 1 vector of initial weights
    error_per_iter = zeros(iterations,1); % error_per_iter(i) records training error in iteration i of GD.
    % dont forget to update these values within the loop!
    step_size = initial_step_size; 
    
    for iter = 1:iterations 
    % FILL IN THE REST OF THE CODE %
    gradient = zeros(1,size(Xtrain,2)); 
        for row = 1:size(Xtrain,1)
            add = Ytrain(row) * Xtrain(row,:) - Xtrain(row,:) / (1 + exp(- weights' * Xtrain(row,:)'));
            gradient = gradient + add; 
        end 
        gradient = gradient'; 
        
        % update weight P * 1 
        step_size = step_size / sqrt(iter); 
        weights = weights + step_size * gradient; 
        
        % P(Y = 1 | x,w) 
        prob_1 = 1 ./ (1 + exp(-weights' * Xtrain')); 
        predicted = prob_1; 
        predicted(predicted > 0.5) = 1; 
        predicted(predicted <= 0.5) = 0; 
        
        diff = predicted' - Ytrain; 
        error_per_iter(iter) = sum(abs(diff))/size(diff,1); 
    end 
end










