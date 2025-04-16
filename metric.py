import numpy as np
from loguru import logger
ignore_index=-100

def ECE(pred, label,M): #pred:[N,k] label:[N]
    conf=np.max(pred,axis=-1)
    y=np.argmax(pred,axis=-1)
    count_M=np.zeros(M)
    count_A=np.zeros(M)
    count_C=np.zeros(M)
    N=len(label)
    for i in range(N):
        if label[i]!=ignore_index:
            k=int(np.floor(conf[i]*M))
            if k==M:
                k=M-1
            count_M[k]+=1
            if y[i]==label[i]:
                count_A[k]+=1
            count_C[k]+=conf[i]
    out=np.sum(np.abs(count_A-count_C))/np.sum(label!=ignore_index)
    sgn=np.sum(count_A-count_C)/np.sum(label!=ignore_index)
    if sgn>0:
        logger.info('underconfidence')
    else:
        logger.info('overconfidence')
    return out

def MCE(pred, label,M): #pred:[N,k] label:[N]
    conf=np.max(pred,axis=-1)
    y=np.argmax(pred,axis=-1)
    count_M=np.zeros(M)
    count_A=np.zeros(M)
    count_C=np.zeros(M)
    N=len(label)
    for i in range(N):
        if label[i]!=ignore_index:
            k=int(np.floor(conf[i]*M))
            if k==M:
                k=M-1
            count_M[k]+=1
            if y[i]==label[i]:
                count_A[k]+=1
            count_C[k]+=conf[i]
    out=np.max(np.abs(count_A-count_C)/(count_M+1e-9))
    return out

def AdaECE(pred, label,M): #pred:[N,k] label:[N]
    conf=np.max(pred,axis=-1)
    y=np.argmax(pred,axis=-1)
    sorted_index=np.argsort(conf)
    cut_index=np.array_split(sorted_index,M)
    count_M=np.zeros(M)
    count_A=np.zeros(M)
    count_C=np.zeros(M)
    N=len(label)
    for i in range(M):
        if label[i]!=ignore_index:
            count_M[i]=len(cut_index[i])
            for j in cut_index[i]:
                if y[j]==label[j]:
                    count_A[i]+=1
                count_C[i]+=conf[j]

    out=np.sum(np.abs(count_A-count_C))/np.sum(label!=ignore_index)
    return out

def Classwise_ECE(pred, label,M): #pred:[N,k] label:[N]
    num_classes=len(pred[0])
    conf=np.max(pred,axis=-1)
    y=np.argmax(pred,axis=-1)
    count_M=np.zeros((M,num_classes))
    count_A=np.zeros((M,num_classes))
    count_C=np.zeros((M,num_classes))
    N=len(label)
    for i in range(N):
        if label[i]!=ignore_index:
            for j in range(num_classes):
                k=int(np.floor(pred[i][j]*M))
                if k==M:
                    k=M-1
                count_M[k][j]+=1
                if j==label[i]:
                    count_A[k][j]+=1
                count_C[k][j]+=pred[i][j]

    out=np.sum(np.abs(count_A-count_C))/(np.sum(label!=ignore_index)*num_classes)
    return out

def Full_ECE(pred, label,M): #pred:[N,k] label:[N]
    num_classes=len(pred[0])
    conf=np.max(pred,axis=-1)
    y=np.argmax(pred,axis=-1)
    count_M=np.zeros(1)
    count_A=np.zeros(M)
    count_C=np.zeros(M)
    N=len(label)
    for i in range(N):
        if label[i]!=ignore_index:
            count_M+=1
            for j in range(num_classes):
                k=int(np.floor(pred[i][j]*M))
                if k==M:
                    k=M-1
                #count_M[k]+=1
                if j==label[i]:
                    count_A[k]+=1
                count_C[k]+=pred[i][j]

    out=np.sum(np.abs(count_A-count_C))/np.sum(count_M)
    return out

def tar_p_ECE(pred, label,tar_prob): #pred:[N,k] label:[N]
    N=len(label)
    count_L1_error=0
    count_kl_error=0
    count_M=0
    for i in range(N):
        if label[i]!=ignore_index:
            count_M+=1
            label_prob=pred[i][label[i]]
            count_L1_error+=np.abs(label_prob-tar_prob[i])
            count_kl_error+=tar_prob[i]*(np.log(tar_prob[i]+1e-8)-np.log(label_prob))+(1-tar_prob[i])*(np.log(1-tar_prob[i]+1e-8)-np.log(1-label_prob))
    
    out_L1_error=count_L1_error/np.sum(label!=ignore_index)
    out_kl_error=count_kl_error/np.sum(label!=ignore_index)

    return out_L1_error,out_kl_error