# coding:utf-8
# created by Phoebe_px on 2017/5/23
from extract_feature_cate8_new import get_trainingset
from extract_feature_cate8_new import get_testset
from sklearn.model_selection import train_test_split
import xgboost as xgb
from extract_feature_cate8_new import evalution

def xgboost_make_submission():
    train_start_date = '2016-02-01'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-02-01'
    sub_end_date = '2016-04-16'
    pair, feature, label = get_trainingset(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(feature.values, label.values, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3,
        'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    param['eval_metric']='logloss'
    #param['eval_metric'] = "auc"
    plst = param.items()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)
    sub_user_index, sub_trainning_data = get_testset(sub_start_date, sub_end_date,)
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.03]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('./sub/result.csv', index=False, index_label=False)



def xgboost_cv():
    train_start_date = '2016-02-01'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-02-01'
    sub_end_date = '2016-04-11'
    sub_test_start_date = '2016-04-11'
    sub_test_end_date = '2016-04-16'
    user_index, training_data, label = get_trainingset(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 4000
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    param['eval_metric'] = 'logloss'
    plst = param.items()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train( plst, dtrain, num_round, evallist)

    sub_user_index, sub_trainning_date, sub_label = get_trainingset(sub_start_date, sub_end_date,sub_test_start_date, sub_test_end_date)
    print(sub_trainning_date.shape)
    test = xgb.DMatrix(sub_trainning_date.values)
    y = bst.predict(test)
    pred = sub_user_index.copy()
    y_true = sub_user_index.copy()
    pred['label'] = y
    y_true['label'] = label
    evalution(pred, y_true)


if __name__ == '__main__':
    xgboost_cv()
    xgboost_make_submission()

