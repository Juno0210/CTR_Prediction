import numpy as np
import pandas as pd
import torch
from sklearn.svm import SVC
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from bayes_opt import BayesianOptimization
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # data processing
    data = pd.read_csv('train.csv')
       
    col = ['C15', 'C16', 'C19', 'C21']
    for col in col:
        percentiles = data[col].quantile(0.98)
        if data[col].quantile(0.98) < 0.5 * data[col].max():
            data[col][data[col] >= percentiles] = percentiles
    
    label='click'
    target = [label]
    dense_features = []
    sparse_features = []
    
    for col in (list(data.columns)):
        if data[col].dtype == "object":
            sparse_features.append(col)
        else:
            dense_features.append(col)
    dense_features.remove(label)
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    dense_features.remove('id')
    dense_features.remove('C17')

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
    mms = MinMaxScaler(feature_range=(0, 1))
    
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    
    train, test = train_test_split(data, test_size=0.2, random_state=2048)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    x = data[dense_features].to_numpy()
    y = data["click"].to_numpy()

    # pre-processing for bayesian optimization
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                      stratify = y,
                                      test_size=0.2,
                                      random_state = 2048)

    X_train_scaled = mms.fit_transform(X_train)
    
    X_test_scaled = mms.transform(X_test)
   
    # Define the black box function to optimize.
    def black_box_function(C):
        # C: SVC hyper parameter to optimize for.
        model = SVC(C = C)
        model.fit(X_train_scaled, y_train)
        y_score = model.decision_function(X_test_scaled)
        f = roc_auc_score(y_test, y_score)
        return f

    pbounds = {"C": [0.1, 20]}
   
    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)

    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    # history = model.fit(train_model_input, train[target].values, batch_size=32, epochs=10, verbose=2, validation_split=0.2)
    pred_ans = model.predict(test_model_input, 256)

    ypred=[]
    for i in pred_ans:
        ypred.append(round(i[0]))

    optimizer = BayesianOptimization(f = black_box_function,
                                 pbounds = pbounds, verbose = 2,
                                 random_state = 4)
    optimizer.maximize(init_points = 5, n_iter = 10)
    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
    print("test Roc accuracy ", round(roc_auc_score(test[target].values,ypred), 4))