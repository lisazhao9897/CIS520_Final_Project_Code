% GMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% total_data=[train_inputs;test_inputs];
% topics=[train_topics; test_topics]; 
train_inputs = train_inputs_gmm; % avoid being rescaled 

train_points=size(train_inputs,1);
test_points=size(test_inputs,1);
% all_points=size(total_data,1);
labeldimension=size(train_labels,2);

clusterNumber=10;

reduced_trainingdata=train_inputs; 
reduced_testingdata=test_inputs; 

[Idx,C,~,~]=kmeans(reduced_trainingdata,clusterNumber,'start','uniform'); 

options=statset('MaxIter',10000);
gm = fitgmdist(reduced_trainingdata,clusterNumber,'RegularizationValue',0.1, 'Options',options); 
result = posterior(gm, reduced_trainingdata);
result_test = posterior(gm, reduced_testingdata);

%assign label on result
trainpost_possibility=result;
testpost_possibility=result_test;

traincluster=zeros(train_points,1);
testcluster=zeros(test_points,1);

for i=1:train_points
    max_poss=max(trainpost_possibility(i,:));    
    index=find(trainpost_possibility(i,:)==max_poss(1));
    traincluster(i)=index;
    
end

for i=1:test_points
    max_poss_test=max(testpost_possibility(i,:));
    index_test=find(testpost_possibility(i,:)==max_poss_test(1));
    testcluster(i)=index_test;
end

%calculate mean labels for each cluster
MeanLabels = zeros(clusterNumber,labeldimension); 
for K=1:clusterNumber
    point=find(Idx==K);
    num=size(point,1);    
    LabelsInEachCluster=zeros(num,labeldimension);
    for i=1:num
        LabelsInEachCluster(i,:)=train_labels(point(i),:);
    end
    MeanLabels(K,:)=mean(LabelsInEachCluster);   
end


%assign mean labels to points in each cluster
predict_labels=zeros(train_points,labeldimension);
for i=1:train_points
    ClusterID=Idx(i);
    predict_labels(i,:)=MeanLabels(ClusterID,:);
end

% predict_labels_test=zeros(test_points,labeldimension);
% for i=1:test_points
%     ClusterID_test=testcluster(i);
%     predict_labels_test(i,:)=MeanLabels(ClusterID_test,:);
% end

pred_testlabels=zeros(test_points,labeldimension);
for i=1:test_points
    distance=zeros(clusterNumber,1);
    for j=1:clusterNumber
        distance(j,1)=norm(reduced_testingdata(i,:)-C(j,:));
    end
    minDistance=min(distance);
    clusterid=find(distance==minDistance);
    pred_testlabels(i,:)=MeanLabels(clusterid(1),:);
end


%pred_labels=predict_labels_test;
pred_labels_gmm = pred_testlabels;

% error_gmm_train = error_metric(predict_labels, train_labels); 

pred_labels = (pred_labels_knn + pred_labels_gmm) /2; 