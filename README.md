# LTGN
Loss Trim Graph-based Network (LTGN), an unsupervised framework based on the interpretability of graph neural networks.

# Installation
The LTGN package is developed based on the Python libraries PyTorch and PyG (PyTorch Geometric) framework, and can be run on GPU (recommend) or CPU. Before installing stVGP, please ensure that PyTorch, and PyG (PyTorch Geometric) are already installed. These dependencies are required for LTGN to function properly, but they are not automatically installed during the installation process to allow greater flexibility.
    
    # Installation
    pip install LTGN
    
    # Requirement
    # scipy
    # numpy
    # pandas
    # scikit-learn
    # pingouin

# Quick-start tutorial
Here, we provide guidance on using the LTGN sample data to help you quickly get started with our method. Here we use Beeline data as an example. The data can be found in the data folder (https://github.com/wzdrgi/LTGN/tree/main/data/Beeline/mDC), and the PPI information can be obtained from the link https://figshare.com/articles/dataset/LTGN_PPI/31570009. 


# Import packages
    import pandas as pd
    import numpy as np
    import torch
    import os
    from sklearn.preprocessing import RobustScaler
    import LTGN

# Set parameters
    seed = 0
    top = 500
    threshold = 0
    epoch = 550
    std_scaler = RobustScaler()
    lr = 0.05
    ln = 5
    m = 'ppi'
    ppthreshold = 0
    ref = None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data and processing
    datafile = 'F:\GitDemo\LTGN\data\Beeline/mDC/ExpressionData.csv'
    celltype_data = pd.read_csv(datafile,index_col=0).T
    order_name  = 'F:/GitDemo/LTGN/data/Beeline/mDC/GeneOrdering.csv'
    order = pd.read_csv(order_name,index_col=0)
    sorted_order = order.sort_values(by='Variance', ascending=False)
    topgene = list(sorted_order.iloc[:top,].index)
    selected = celltype_data.loc[:,topgene]
    numeric_cols = selected.select_dtypes(include=['number']).columns
    selected[numeric_cols] = std_scaler.fit_transform(selected[numeric_cols])
    filename = 'mDC' + str(threshold)+ 'seed' + str(seed) + str(epoch) +' ln' + str(ln) + 'top' + str(top) + 'ppth' + str(ppthreshold)+ m + '.csv'

# Run
    edgee_df = LTGN.oneFeaturemodel_ppi(
                            data_df = selected,
                            lr = 0.05,
                            epoch = 1,
                            # Please modify the training epoch.
                            threshold = threshold,
                            ppthreshold = ppthreshold,
                            seed = seed,
                            ln = ln,
                            ref = ref,
                            running_name = 'mDC' + '_rb' + '_top' + str(top) + '_ppth' + str(ppthreshold) + '_seed' + str(seed),
                            m = m,
                            batch_size = None,
                            device = device,
                            spe = 'homo')
    
