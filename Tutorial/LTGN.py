import numpy as np
import pandas as pd
import pingouin as pg
import copy
import time
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.nn import SAGEConv
from scipy import stats
from sklearn.preprocessing import MaxAbsScaler

def getCorelationship(n,da):
    coor = np.corrcoef(da.T)
    corr = []
    for i in range(n):
        for j in range(n):
            if i != j:
                corr.append([coor[i,j], [i, j]])
    return corr

def getCorEdge(c0, threshold):

    coor_array = []
    coor_edge = []

    for i in range(len(c0)):
        coor_array.append(c0[i][0])
        coor_edge.append(c0[i][1])

    coor_array = np.array(coor_array)
    abs_coor = np.abs(coor_array)
    mask = abs_coor >= threshold
    edge_array = np.array(coor_edge)

    corSaveList = edge_array[mask].tolist()
    corDelList = edge_array[~mask].tolist()

    return corSaveList, corDelList

def confusion_matrix(predictionlist, 
                     truelist, 
                     n):
    """
    Calculates classification metrics for link prediction.
    Predicted data contains only positive instances (edges). 
    The 'truelist' contains only existing edges (ground truth).
    """
    save = len(predictionlist)
    real = len(truelist)

    # Calculate True Positives (TP)
    tp = 0
    for i in predictionlist:
        if i in truelist:
            tp = tp + 1
    tpr = tp / real
    if tp == 0:
        print('All predictions are incorrect')
        fp = save - tp
        fpr = fp / (n * (n - 1) - real)
        # Calculate False Negatives (FN)
        fn = real - tp
        # Calculate True Negatives (TN)
        tn = n * (n - 1) - real - fp
        specificity = tn / (n * (n - 1) - real)
        # Calculate Accuracy
        acc = (tp + tn) / (n * (n - 1))
        pres = 0
        # Calculate Precision
        # pres = tp / save
        # Calculate Recall
        # recall = tp / (tp + fn)
        # Calculate F1-score
        # f1 = 2 * pres * recall / (pres + recall)
        ma = 0
        recall = 0

    else:
        # Calculate False Positives (FP)
        fp = save - tp
        fpr = fp/(n * (n - 1)-real)
        # Calculate False Negatives (FN)
        fn = real - tp
        # Calculate True Negatives (TN)
        tn = n * (n - 1) - real - fp
        specificity = tn / (n * (n - 1) - real)
        # Calculate Accuracy
        acc = (tp + tn) / (n * (n - 1))
        # Calculate Precision
        pres = tp / save
        # Calculate Recall
        recall = tp / real
        # Calculate F1-score
        f1 = 2 * pres * recall / (pres + recall)

        print('True Positive Rate (TPR):', tpr, '\nFalse Positive Rate (FPR):', fpr, '\nPrecision:', pres, '\nTrue Negative Rate (TNR)', specificity, '\nacc', acc, '\nf1score', f1, '\nrecall ', recall,'\nsavelen ', save)
    return tpr,fpr,acc,pres

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))
    
class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)

def mymodel(seed,
            rawdata,
            name,
            newedge,
            epochtimes,
            learning_rate,
            gene_num,
            savefilter,
            ln, 
            device):
    
    """
    rawdata: Expanded dataset
    name: Name for storing the dataset
    newedge: Modified edge set
    epochtimes: Number of training epochs
    lr: Learning rate
    lossfunction: Loss function, str
    """

    # Create PyG (PyTorch Geometric) dataset
    class MyOwnDataset(InMemoryDataset):
        def __init__(self, root,transform=None, pre_transform=None, pre_filter=None):
            super(MyOwnDataset, self).__init__(root, transform, pre_transform, pre_filter)
            if not os.path.exists(self.processed_paths[0]):
                self.process()
                
            self.data, self.slices = torch.load(self.processed_paths[0],map_location='cpu')
            
        # def collate(self, data_list):
        #     return Batch.from_data_list(data_list,self.batch_size)
            
        @property
        def raw_file_names(self):
            return []

        @property
        def processed_file_names(self):
            return [name]

        def download(self):
            # Download to `self.raw_dir`.
            pass

        def process(self):
            # Read data into huge `Data` list.
            data_list = []
            indexedge = []
            for ed in newedge:
                e1 = ed
                e2 = [ed[1],ed[0]]
                indexedge.append(e1)
                indexedge.append(e2)
            index = np.transpose(indexedge)
            
            for i in rawdata:
                ex = i
                ee = [[j] for j in ex]
                x = torch.tensor(ee, dtype=torch.float32)
                
                index = torch.tensor(index, dtype=torch.long)
                

                data = Data(x=x, y=x, edge_index=index)
                
                data_list.append(data)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

            self.data, self.slices = self.collate(data_list)
            torch.save((self.data, self.slices), self.processed_paths[0])
            print("Data saved at:", self.processed_paths[0])

    mydataset = MyOwnDataset(root=savefilter)

    class GNN(nn.Module):
        def __init__(self,actf):
            super(GNN, self).__init__()
            torch.manual_seed(seed)
            self.conv1 = SAGEConv(1, 1,aggr='mean')  
            self.conv2 = SAGEConv(1, 1,aggr='mean')
            self.conv3 = SAGEConv(1, 1,aggr='mean')
            self.den = nn.Linear(10, 105)
            self.activation  = actf

        def forward(self, x, edge_index):
            x = x.to(torch.float32)
            h1 = self.conv1(x, edge_index) 
            h1 =self.activation(h1)
            h2 = self.conv2(h1, edge_index)
            h2 = self.activation(h2)
            return h2
        
    #train
    activation_function = {'relu':nn.ReLU(),'sigmoid':nn.Sigmoid(),'tanh':nn.Tanh(),'Leakrelu':nn.LeakyReLU(),'elu':nn.ELU()}
    
    model = GNN(activation_function['sigmoid'])
    model.to(device)

    lf_list = {'XT':XTanhLoss(),'L1M':nn.L1Loss(reduction='mean'),'L1S':nn.L1Loss(reduction='sum'),'SL1': nn.SmoothL1Loss(beta=1),'LC':LogCoshLoss(),'XS':XSigmoidLoss(),'MSE':nn.MSELoss()}

    loss_function = lf_list['MSE']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    def train(dataset,model):
        model.train()
        loss_all = 0
        for data in dataset:
            data.to(device)
            
            optimizer.zero_grad()
            x = data.x
            x.to(device)
            edgeindex = data.edge_index
            edgeindex.to(device)
            
            output = model(x, edgeindex)
            real = data.y
            real.to(device)

            loss1 = loss_function(output, real)
            loss = loss1

            loss.backward()
            loss_all += loss.item()

            optimizer.step()

        return loss_all / len(dataset)
    
    ll = []

    for i in range(epochtimes):
        model.train()
        optimizer.zero_grad()
        losses = train(mydataset,model)
        if torch.is_tensor(losses):
            losses = losses.detach().cpu().item()
        ll.append(losses)
        
    bbbb = np.mean(ll[-ln:])
    return bbbb

def datamodel(data,
              save_edge,
              epochtimes,
              lr,
              datafilter_name,
              goldenstand,
              ln,
              seed,
              device):
    """
    :param data
    :param edge
    """

    datasetname = datafilter_name
    ga = goldenstand
    start = time.process_time()
    d = data
    # 105 * 10
    gene_num = len(d[0])
    data = np.concatenate((d, d), axis=1) 
    # 105 * 20
    saveEdge = []
    for i in save_edge:
        e1 = [i[0], i[1] + gene_num]
        saveEdge.append(e1)

    now = len(saveEdge) 
    saveEdge1 = copy.deepcopy(saveEdge) 
    
    basic = mymodel(seed, 
                    data,
                    'basic',
                    saveEdge1,
                    epochtimes,
                    lr,
                    gene_num,
                    datasetname,
                    ln,
                    device)
    
    print('basic finish')
    
    outedge = []
    # loop = tqdm(saveEdge, total=now)
    # for edge in saveEdge:
    for j in range(now):
        ee = saveEdge[j]
        inn = ee[0]
        ouu = ee[1]
        s1 = time.process_time()
        saveEdge1.remove(ee)
        if len(saveEdge1) == 0:
            break
        loss50 = mymodel(seed,
                         data, 
                         '{}'.format([inn,ouu]), 
                         saveEdge1, 
                         epochtimes, 
                         lr, 
                         gene_num, 
                         datasetname,
                         ln,
                         device)
        if loss50 <= basic:
            basic = loss50
            print('Deleted:', ee, loss50)
        else:
            saveEdge1.append(ee)
            outedge.append(ee)
            print('Saved:', ee, loss50)
        end1 = time.process_time()

    gs = []
    for i in ga:
        out = i[0]
        input = i[1] + gene_num
        gs.append([out, input])
    #print('GS len', len(gs))
    tpr,fpr,ac,recall = confusion_matrix(outedge, gs, gene_num)
    print('save', len(outedge))
    end = time.process_time()
    print('time', end - start)
    outcome = ['True Positive Rate{}'.format(tpr),'\nFalse Positive Rate{}'.format(fpr),'\nRecall{}'.format(recall),'\nAccuracy{}'.format(ac),'\nSaved Length',len(outedge)]
    return tpr,fpr,recall,ac,outedge

def allList(n):
    return [[i, j] for i in range(n) for j in range(n) if i != j]

def getPcorship(name,
                da,
                m,
                ref):
    
    n = len(name)
    df = pd.DataFrame(data=da,
                      columns=name)
    pcorr_list = []
    
    alist = allList(n)

    for e in alist:
        i = e[0]
        j = e[1]
        genename = copy.deepcopy(name)
        genename.remove(name[i])
        genename.remove(name[j])
        pc=pg.partial_corr(data=df,x=name[i],y=name[j],covar=genename,method='pearson')
        pc = np.array(pc).tolist()[0]
        pccorr = pc[1]
        pccorr_p = pc[-1]
        p = [pccorr,pccorr_p,[i,j]]
        pcorr_list.append(p)

    if ref == 'pp':
        pcorSaveList = [i[-1] for i in  pcorr_list if abs(i[1]) < m]
    elif ref == 'pc':
        pcorSaveList = [i[-1] for i in  pcorr_list if abs(i[0]) >= m]
    
    if ref != 'pc' and ref != 'pp':
        raise ValueError("Invalid relationship type. Please use a supported correlation evaluation metric, such as ref = 'pc' for Pearson correlation selection.")

    pcorDeleList = []
    
    for i in alist:
        if i not in pcorSaveList:
            pcorDeleList.append(i)

    return pcorSaveList,pcorDeleList

def onlyData(data, 
             gs, 
             threshold, 
             lr, 
             epoch, 
             datasetname, 
             ln, 
             seed, 
             genename,
             relationship='c',
             ref = 'pc',
             device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    
    gene_number3 = len(data[0])

    if relationship == 'c':
        c0 = getCorelationship(gene_number3, da=data)
        save, dele = getCorEdge(c0, threshold)

    if relationship == 'p':
        save, dele = getPcorship(name = genename,
                                 data = data,
                                 m = threshold,
                                 ref = ref)
        
    if relationship != 'c' and relationship != 'p':
        raise ValueError("Invalid relationship type. Please use a supported correlation evaluation metric, such as 'c' for Pearson correlation.")
    
    if len(save) != 0:
        orr = confusion_matrix(save, gs, gene_number3)
        print('Begin run!')
        tpr,fpr,recall,ac,edgee = datamodel(data, 
                                            save, 
                                            epoch, 
                                            lr, 
                                            datasetname, 
                                            gs, 
                                            ln, 
                                            seed,
                                            device)
        print('Done!')
    else:
        raise ValueError(
        "Failed to generate the initial graph. The threshold may be too strict to retain any edges."
        "Please try using a lower threshold value and run again.")
    
    return orr,tpr,fpr,recall,ac,len(gs),edgee

def dele_edge(edge,sets):
    i = edge[0]
    j = edge[1]
    indexx = 0
    for gene_set in sets:
        if i in gene_set and j in gene_set:
            indexx = indexx+1
            
    return indexx

def oneFeature(
        data,
        edge,
        epochtimes,
        learning_rate,
        datalen,
        seed,
        ln,
        device,
):
    indexedge = []
    for ed in edge:
        e1 = ed
        e2 = [ed[1], ed[0]]
        indexedge.append(e1)
        indexedge.append(e2)
    indexedge = np.transpose(indexedge)
    indexedge = torch.tensor(indexedge, dtype=torch.long)
    data = Data(x=data, y=data, edge_index=indexedge)

    class GCN(nn.Module):
        def __init__(self):
            super(GCN, self).__init__()
            torch.manual_seed(seed)
            self.conv1 = SAGEConv(datalen, datalen)  
            self.conv2 = SAGEConv(datalen, datalen)
            self.conv3 = SAGEConv(datalen, datalen)

        def forward(self, x, edge_index):
            x = x.to(torch.float32)
            h1 = self.conv1(x, edge_index)  
            h1 = F.relu(h1)
            h2 = self.conv2(h1, edge_index)
            h2 = F.relu(h2)

            return h2
    
    model = GCN()
    model.to(device)
    loss_function = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    def train(dataset,
              model):    
        optimizer.zero_grad()
        dataset.to(device)
        out = model(dataset.x, dataset.edge_index)
        loss = loss_function(out, dataset.y)
        loss.backward()
        optimizer.step()
        return loss,out
    
    ll = []

    for i in range(epochtimes):
        loss,out = train(data,model)
        if loss.is_cuda:
            loss_item = loss.detach().cpu().item()
        else:
            loss_item = loss.detach().item()
        ll.append(loss_item)
        bbbb = np.mean(ll[-ln:])

    return bbbb,len(ll)

def oneFeaturemodel_getCorelationship(n,
                                      da
):
    """
    Calculate the correlation between genes and the corresponding p-values.
    :param n: Number of genes
    :param da: Gene expression data
    :return: Correlation results, where the first element is the correlation coefficient, 
             the second is the p-value, and the third is the corresponding edge.
    """

    corr = []
    for i in range(n):
        for j in range(n):
            if i != j:
                a = [m[i] for m in da]
                b = [n[j] for n in da]
                coo = stats.pearsonr(a, b)
                co = coo[0]
                cop = coo[1]
                corr.append([co, cop, [i, j]])
    return corr

def oneFeaturemodel_getCorEdge(
    cor, 
    m, 
    ref    
):
    """
    通过p值判断两个基因有相关关系是否合理
    :param cor: 相关性和p值列表
    :param m: 判断标准
    :return: 具有相关性和不具有相关性的边
    """
    cor_save = []
    cor_del = []

    if ref == 'pc':
        for i in cor:
            if abs(i[0]) >= m:
                cor_save.append(i)
            else:
                cor_del.append(i)
                
    if ref == 'pp':
        for i in cor:
            if abs(i[1]) <= m:
                cor_save.append(i)
            else:
                cor_del.append(i)

    #cor_save = sorted(cor_save,key=(lambda x:abs(x[0])),reverse=True)
    corSaveList = [i[2] for i in cor_save]  
    corDelList = [i[2] for i in cor_del]  
    return corSaveList, corDelList

def oneFeaturemodel(
        data,
        goldens,
        lr,
        epochtimes,
        threshold,
        seed,
        ln,
        ref,
        gene_set = None,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
):
    
    # data sample * gene
    geneNumber = len(data[0])
    datalen = len(data)
    
    c0 = oneFeaturemodel_getCorelationship(geneNumber, data)

    f_save_list, f_del_list = oneFeaturemodel_getCorEdge(c0, 
                                                         threshold, 
                                                         ref)

    if gene_set != None:
        save = []
        for edge in f_save_list:
            score = dele_edge(edge,gene_set)
            if score == 0:
                save.append(edge)
            else:
                print('dele edge',edge)
        
        if len(save) != 0:
            print('real save len',len(save))
            da = np.concatenate((data, data), axis=1)
            da = da.T
            x = torch.tensor(da, dtype=torch.float)
            saveEdge = []
            for i in save:
                e1 = [i[0], i[1] + geneNumber]
                saveEdge.append(e1)
            now = len(saveEdge)  
            saveEdge1 = copy.deepcopy(saveEdge)  

            basic = oneFeature(x, 
                               saveEdge1, 
                               epochtimes, 
                               lr,
                               datalen,
                               seed,
                               ln,
                               device)
            
            print('basic finish')
            outedge = []
            c = 0
            # loop = tqdm(saveEdge, total=now)
            # for edge in saveEdge:
            for j in range(now):
                ee = saveEdge[j]
                inn = ee[0]
                ouu = ee[1]
                s1 = time.process_time()
                saveEdge1.remove(ee)
                loss50 = oneFeature(x, 
                                    saveEdge1, 
                                    epochtimes, 
                                    lr,
                                    datalen,
                                    seed,
                                    ln,
                                    device)
                c = c + 1
                if loss50 <= basic:
                    basic = loss50
                    print('Right! Delete!', ee, loss50)
                else:
                    saveEdge1.append(ee)
                    outedge.append(ee)
                    print('Wrong! Save!', ee, loss50)
                end1 = time.process_time()
                print('Time taken to process edge {}:'.format(ee), end1 - s1)

            gs = []
            for i in goldens:
                out = i[0]
                input = i[1] + geneNumber
                gs.append([out, input])
            print('GS len', len(gs))
            tpr, fpr, ac, pres = confusion_matrix(outedge, gs, geneNumber)
            print('save', len(outedge))
            outcome = [
            'True Positive Rate: {}'.format(tpr),
            'False Positive Rate: {}'.format(fpr),
            'Accuracy: {}'.format(ac),
            'Number of retained edges: {}'.format(len(outedge))]
            print(outcome)

        else:
            raise ValueError(
                "Failed to generate the initial graph. This may be caused by one of the following reasons:\n"
                "1. The threshold is too strict, resulting in no edges retained.\n"
                "2. Regulatory relationships could not be found even within the same pathway (e.g., no edges exist among pathway genes).\n"
                "Please try using a lower threshold value or check the pathway data and relationship definitions."
            )
        
        return tpr,fpr,ac,outedge,pres

    if gene_set == None:

        da = np.concatenate((data, data), axis=1)
        da = da.T
        x = torch.tensor(da, dtype=torch.float32)
        saveEdge = []
        for i in f_save_list:
            e1 = [i[0], i[1] + geneNumber]
            saveEdge.append(e1)
        
        now = len(saveEdge) 

        saveEdge1 = copy.deepcopy(saveEdge) 

        basic = oneFeature(x, 
                        saveEdge1, 
                        epochtimes, 
                        lr,
                        datalen,
                        seed,
                        ln,
                        device)
        
        print('basic finish')

        outedge = []
        c = 0

        for j in range(now):
            ee = saveEdge[j]
            inn = ee[0]
            ouu = ee[1]
            s1 = time.process_time()
            saveEdge1.remove(ee)
            loss50 = oneFeature(x, 
                                saveEdge1, 
                                epochtimes, 
                                lr, 
                                datalen, 
                                seed, 
                                ln,
                                device)
            c = c + 1
            if loss50 <= basic:
                basic = loss50
                print('Right! Delete!', ee, loss50)
            else:
                saveEdge1.append(ee)
                outedge.append(ee)
                print('Wrong! Save!', ee, loss50)
            end1 = time.process_time()
            # Print the time taken to process edge {ee}
            print('Time taken to process edge {}:'.format(ee), end1 - s1)
        
        gs = []

        for i in goldens:
            out = i[0]
            input = i[1] + geneNumber
            gs.append([out, input])

        print('GS len', len(gs))
        tpr, fpr, ac, mm = confusion_matrix(outedge, gs, geneNumber)
        print('save', len(outedge))
        outcome = [
        'True Positive Rate: {}'.format(tpr),
        'False Positive Rate: {}'.format(fpr),
        'Accuracy: {}'.format(ac),
        'Number of retained edges: {}'.format(len(outedge))]
        print(outcome)
        return tpr,fpr,ac,outedge,pres

def getCorelationship_ppi(da):

    n = da.shape[1]  
    corr = np.corrcoef(da, rowvar=False)  
    correlations = []

    for i in range(n):
        for j in range(n):
            if i != j:
                correlations.append([[i, j], corr[i, j]])
    return correlations

def getCorEdge_ppi(cor, m, ref):
    """
    Determine whether the correlation between two genes is significant based on p-values.
    :param cor: List of correlation coefficients and p-values
    :param m: p-value significance threshold
    :return: Edges with significant correlation and edges without significant correlation
    """
    cor_save = []
    or_edge = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]]
    cor_del = []
    # 阈值
    for i in cor:
        if abs(i[1]) >= m:
            if i[0] in or_edge:
                pass

            else:
                cor_save.append(i)
        else:
            cor_del.append(i)
    #cor_save = sorted(cor_save,key=(lambda x:abs(x[0])),reverse=True)
    corSaveList = [i[0] for i in cor_save]  
    corDelList = [i[0] for i in cor_del]  
    return corSaveList, corDelList

def oneFeaturemodel_ppi(
        data_df,
        lr,
        epoch,
        threshold,
        ppthreshold,
        seed,
        ln,
        ref,
        running_name,
        m,
        batch_size,
        gene_relation_path = None,
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        spe = None,
):
    edgefile = running_name + '.txt'
    data_df.drop(columns=data_df.columns[(data_df == 0).all()], inplace=True)
    genename = data_df.columns.tolist() #genename
    geneNumber = data_df.shape[1]

    scaler = MaxAbsScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)

    nm_data = data_scaled.to_numpy()
    datalen = len(nm_data)

    if m == 'ppi+cor':
        ppilinks = pd.read_csv('ppi_geen_relation.csv')
        co = getCorelationship_ppi(nm_data)
        f_save_list, f_del_list = getCorEdge_ppi(co, threshold,'cc')
        print(running_name,len(f_save_list))
        links_df = pd.DataFrame(f_save_list, columns=['gene1', 'gene2'])
        links_df['gene1'] = links_df['gene1'].apply(lambda x: genename[x])
        links_df['gene2'] = links_df['gene2'].apply(lambda x: genename[x])

        
        filtered_links = pd.merge(links_df, ppilinks, on=['gene1','gene2'])
        filtered_links.to_csv(running_name+'-'+'filtered_link_c1_'+str(threshold)+'.csv')

        gene_to_index = {gene: index for index, gene in enumerate(genename)}
        filtered_links['gene1'] = filtered_links['gene1'].apply(lambda x: gene_to_index.get(x, -1))
        filtered_links['gene2'] = filtered_links['gene2'].apply(lambda x: gene_to_index.get(x, -1))   
        fl = filtered_links[['gene1','gene2']]
        linky = fl.values.tolist()
        edge = linky
        # linky = [i.split() for i in linky]
        saveEdge = []
        for i in edge:
            e1 = [i[0], i[1] + geneNumber]
            saveEdge.append(e1)

    if m == 'ppi':
        if spe =='homo':
            if gene_relation_path != None:
                gene_relation = pd.read_csv(gene_relation_path,index_col=0)
            else:
                # # Replace the PPI information path
                gene_relation = pd.read_csv('F:/LWY工作/Code and data/Data/Beeline/homo_ppigene_relation.csv',index_col=0)
                
        elif spe=='mouse':
            if gene_relation_path != None:
                gene_relation = pd.read_csv(gene_relation_path,index_col=0)
            else:
                # # Replace the PPI information path
                gene_relation = pd.read_csv('F:/LWY工作/Code and data/Data/Beeline/mouse_ppi_gene_relation.csv',index_col=0)

        elif spe =='mouse_l':
            if gene_relation_path != None:
                gene_relation = pd.read_csv(gene_relation_path,index_col=0)
            else:
                # # Replace the PPI information path
                gene_relation = pd.read_csv('F:/LWY工作/Code and data/Data/Beeline/mouse_ppi_gene_relation_l.csv',index_col=0)
        
        real_relation = gene_relation[gene_relation['score'] >= ppthreshold]
        print('filterd ppi likns',real_relation.shape)
        filtered_df = real_relation[real_relation['gene1'].isin(genename) & real_relation['gene2'].isin(genename)]

        print(filtered_df.shape)
        gene_to_index = {gene: index for index, gene in enumerate(genename)}

        filtered_df['gene1'] = filtered_df['gene1'].map(gene_to_index)
        filtered_df['gene2'] = filtered_df['gene2'].map(gene_to_index)
        links = filtered_df[['gene1','gene2']]
        print('use edge',links.shape)
        linky = links.values.tolist()
        
        edge = linky
        saveEdge = []
        for i in edge:
            e1 = [i[0], i[1] + geneNumber]
            e2 = [i[1],i[0]+geneNumber]
            saveEdge.append(e1)
            saveEdge.append(e2)
        print('need to process',len(saveEdge))

    if m == 'cor':
        co = getCorelationship_ppi(nm_data)
        f_save_list, f_del_list = getCorEdge_ppi(co, threshold, ref)
        print(running_name,len(f_save_list))
        saveEdge = []
        for i in f_save_list:
            e1 = [i[0], i[1] + geneNumber]
            saveEdge.append(e1)
    
    da = np.concatenate((nm_data, nm_data), axis=1)
    da = da.T
    x = torch.tensor(da, dtype=torch.float)
    now = len(saveEdge)  
    saveEdge1 = copy.deepcopy(saveEdge)  
    basic,losslen = oneFeature(x, 
                               saveEdge1, 
                               epoch, 
                               lr,
                               datalen,
                               seed,
                               ln,
                               device)
    print('basic finish',basic)
    outedge = []

    st = time.time()
    c = 0
    for j in range(now):
        ee = saveEdge[j]
        inn = ee[0]
        ouu = ee[1]
        
        saveEdge1.remove(ee)
        
        loss50,losslen = oneFeature(x, 
                               saveEdge1, 
                               epoch, 
                               lr,
                               datalen,
                               seed,
                               ln,
                               device)
        
        print('loss50',loss50)
        runt = time.time()
        
        c = c + 1
        if loss50 <= basic:
            basic = loss50
            en = time.time()
            if j % 100 == 0:
                with open(edgefile,'a') as f:
                    f.write('\n'+'Correctly deleted\t'+str(ee)+str(len(outedge))+'-'+'\tloss'+str(loss50)+'-'+str(losslen)+'--time'+str(en-st)+'\n')

            print('Correctly deleted', ee, loss50,'time',runt-st)
            print(len(saveEdge1))
        else:
            saveEdge1.append(ee)
            outedge.append(ee)
            en = time.time()
            if j % 100 == 0:
                with open(edgefile,'a') as f:
                    f.write('\nsave\t'+str(ee)+'\t'+str(len(outedge))+'-'+str(loss50)+'--'+str(en-st)+'\n')
            print('save', ee, loss50,'time',runt-st)
  
    edgee = []
    for i in outedge:
        out = i[0]
        input = i[1] -geneNumber
        edgee.append([out, input])

    edgee_df = pd.DataFrame(edgee,columns=['gene1','gene2'])
    edgee_df['gene1'] = edgee_df['gene1'].apply(lambda x: genename[x])
    edgee_df['gene2'] = edgee_df['gene2'].apply(lambda x: genename[x])
    
    return edgee_df
    

