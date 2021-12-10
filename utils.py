import torch
import time
import glob
import numpy as np
import torch.nn.functional as F

def save_model(model,args,path,is_best):
    torch.save(model.state_dict(),args.save_path+path)
    if is_best:
        torch.save(model.state_dict(),args.save_path+'best.pth')

def MAE(predict,age):
    predict[predict>=0.5] = 1
    predict[predict<0.5] = 0
    predict_age = torch.sum(predict,dim=1)[:,0] + 15
    abs_error = torch.sum(torch.abs(predict_age - age))
    mean_abs_error = abs_error/len(age)
    return mean_abs_error

def make_task_importance(data_path):
    lambda_t = []
    age_list = glob.glob(data_path+'\*')
    for age in age_list:
        temp_list = glob.glob(age+'\*')
        lambda_t.append(len(temp_list))
    lambda_t = np.sqrt(lambda_t)
    summary = np.sum(lambda_t)
    lambda_t = lambda_t / summary
    return torch.tensor(lambda_t)

def importance_cross_entropy(predict,label,importance):
    predict = torch.log(predict)
    entropy = torch.sum(-1*predict*label,dim=2)
    entropy = entropy * importance
    loss = torch.sum(entropy) / label.shape[0]
    return loss
    
def make_label(age):
    batch_size = age.shape[0]
    k = torch.tensor(torch.arange(15,72).repeat(batch_size).reshape(batch_size,-1))
    label = k-age.reshape(batch_size,-1)
    label[label>=0], label[label<0] = 0, 1
    label = label.reshape(batch_size,72-15,1)
    true = torch.cat((label,1-label),dim=2)
    return true