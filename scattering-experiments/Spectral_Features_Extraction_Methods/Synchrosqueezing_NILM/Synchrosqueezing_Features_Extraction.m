clear all
clc
close all

root_address = '/Users/evertonluizdeaguiar/Documents/Doutorado_PC_Linux/Artigo_Sensors_2023/Metodos_Para_Testar/Synchrosqueezing'
data_address = '/Users/evertonluizdeaguiar/Documents/Doutorado_PC_Linux/Artigo_Sensors_2023/Metodos_Para_Testar/Synchrosqueezing/data.mat'

newData1 = load(data_address);

% Create new variables in the base workspace from those fields.
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end

clear newData1

% Decomposing the signals using Synchrosqueezing
n_train = size(x_train,1);
n_test = size(x_test,1);

% for each shample, extract the features by means of Synchrosqueezing
% technique
S = [];
W = [];
N = [];
cont=0;
kk=1;
for k=4:n_train
    
    [s,w,n] = fsst(x_train(k,:));
    Stemp = [];
    for kkk=1:size(s,1) % for each sub-band
       
        s_splits = buffer(s(kkk,:), numel(s(kkk,:))/5); % Split into five grids
        
        % s_splits.shape = (5,3328)
        
        % now apply the average of the coefficients
        
        s1 = mean(s_splits,1); % this is a complex number
        s1 = abs(s1); % take the absolute value
        
        % s.shape = (1,5)
        
        Stemp = [Stemp; s1];
        
    end
    
    
    S{kk} = Stemp; % S.shape = (n_freq, n_grids)
    W{kk} = w;
    N{kk} = n;
       
    if kk==100
       cont = cont+1;
       kk=1;
       
       % Save File
       save(strcat('FSST_Train_',num2str(cont)),'S','W','N')

       % Clear variables
       S = []
       W = []
       N = []  
    else
       kk = kk+1;       
    end
    
        
end

save(strcat('FSST_Train_',num2str(cont)),'S','W','N')

% Clear all Variables
S = [];
W = [];
N = [];

cont=0;
kk=1;
for k=1:n_test
    
    [s,w,n] = fsst(x_test(k,:));
    Stemp = [];
    for kkk=1:size(s,1) % for each sub-band
       
        s_splits = buffer(s(kkk,:), numel(s(kkk,:))/5); % Split into five grids
        
        % s_splits.shape = (5,3328)
        
        % now apply the average of the coefficients
        
        s1 = mean(s_splits,1); % this is a complex number
        s1 = abs(s1); % take the absolute value
        
        % s.shape = (1,5)
        
        Stemp = [Stemp; s1];
        
    end
    
    
    S{kk} = Stemp; % S.shape = (n_freq, n_grids)
    W{kk} = w;
    N{kk} = n;
       
    if kk==100
       cont = cont+1;
       kk=1;
       
       % Save File
       save(strcat('FSST_Test_',num2str(cont)),'S','W','N')

       % Clear variables
       S = []
       W = []
       N = []  
    else
       kk = kk+1;       
    end
    
        
end

save(strcat('FSST_Test_',num2str(cont)),'S','W','N')