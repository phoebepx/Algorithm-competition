# coding:utf-8
# created by Phoebe_px on 2017/5/23
'''
cleaning data
make feature
'''
import pandas as pd
import pickle
import os
from datetime import datetime,timedelta
from sklearn import preprocessing
import math
import numpy as np
datapath='./data/'
fname_action2='JData_Action_201602.csv'
fname_action3='JData_Action_201603.csv'
fname_action4='JData_Action_201604.csv'
fname_user='JData_User.csv'
fname_product='JData_Product.csv'
fname_comment='JData_Comment.csv'
interimvar='./temp/'
comment_date = ['2016-02-01', '2016-02-08', '2016-02-15', '2016-02-22', '2016-02-29', '2016-03-07', '2016-03-14','2016-03-21', '2016-03-28','2016-04-04', '2016-04-11', '2016-04-15']

# 2,3,4月所有日期列表
def get_all_date():
    path = interimvar+'all_date.pkl'
    if os.path.exists(path):
        return_list=pickle.load(open(path,'rb'))
    else:
        return_list=[]
        return_list.append("2016-01-31")#二月份包括1/31
        year="2016-"
        month="02-"
        for day in range(1,30):
            if day < 10:
                day="0"+str(day)
            else:
                day=str(day)
            date=year+month+day
            return_list.append(date)
        month = "03-"
        for day in range(1, 32):
            if day < 10:
                day = "0" + str(day)
            else:
                day = str(day)
            date = year + month + day
            return_list.append(date)
        month = "04-"
        for day in range(1, 17):
            if day < 10:
                day = "0" + str(day)
            else:
                 day = str(day)
            date = year + month + day
            return_list.append(date)
        pickle.dump(return_list, open(path, 'wb'))
    return return_list
# 时间index
def date_index(sdate):
    alldate=get_all_date()
    dateindex=alldate.index(sdate.split(' ')[0])
    return dateindex
# 截取time日期
def get_day(sdate):
    return sdate.split(' ')[0]

# pandas所有列get_dummies
def get_dummies_for_pd(df):
    name = list(df.columns)
    for i in name:
        temp=pd.get_dummies(df[i],prefix=i)
        df=pd.concat([df,temp],axis=1)
        del df[i]
    return df

# pandas 每一列归一化到0,1区间
def norm_df(df):
    from sklearn import preprocessing
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled,index=df.index,columns=df.columns)
    return df

#用户表处理
def change_age(age):
    if age == u'-1':
        return 0
    elif age == u'15岁以下':
        return 1
    elif age == u'16-25岁':
        return 2
    elif age == u'26-35岁':
        return 3
    elif age == u'36-45岁':
        return 4
    elif age == u'46-55岁':
        return 5
    else:
        return 6

def deal_user():
    path = interimvar+'clean_user.pkl'
    if os.path.exists(path):
        clean_user=pickle.load(open(path,'rb'))
    else:
        unclean_user=pd.read_csv(datapath+fname_user, encoding='gbk')
        #删除age,sex为空的用户,并离散化处理
        unclean_user=unclean_user[unclean_user['age'].notnull()]
        unclean_user['age']=unclean_user['age'].map(change_age)
        unclean_user['sex']=unclean_user['sex'].astype(int)
        age = pd.get_dummies(unclean_user['age'], prefix='age')
        sex = pd.get_dummies(unclean_user['sex'],prefix='sex')
        user_lv = pd.get_dummies(unclean_user['user_lv_cd'],prefix='user_lv_cd')
        #删除注册时间在4/15之后的，并归一化处理
        unclean_user['user_reg_tm']=pd.to_datetime(unclean_user['user_reg_tm'])
        unclean_user['time_delta']= pd.Series([x.days for x in (unclean_user['user_reg_tm'] - datetime(2016,4,15))])
        unclean_user=unclean_user[unclean_user['time_delta']<0]
        unclean_user['time_delta']=pd.Series(preprocessing.scale(unclean_user['time_delta'].apply(abs).values))
        clean_user=pd.concat([unclean_user['user_id'],age,sex,user_lv,unclean_user['time_delta']],axis=1)
        pickle.dump(clean_user, open(path, 'wb'))
    return clean_user

#商品表处理
def deal_product():
    path = interimvar+'clean_product_cate8.pkl'
    if os.path.exists(path):
        clean_product=pickle.load(open(path,'rb'))
    else:
        unclean_product = pd.read_csv(datapath+fname_product, encoding='gbk')
        a1 = pd.get_dummies(unclean_product['a1'], prefix='a1')
        a2 = pd.get_dummies(unclean_product['a2'],prefix='a2')
        a3 = pd.get_dummies(unclean_product['a3'],prefix='a3')
        clean_product=pd.concat([unclean_product[['sku_id']],a1,a2,a3],axis=1)
        pickle.dump(clean_product, open(path, 'wb'))
    return clean_product

#2,3,4月行为数据汇总,cate=8
def gather_action():
    path = interimvar+'all_action_cate8.pkl'
    if os.path.exists(path):
        all_action=pickle.load(open(path,'rb'))
    else:
        action2=pd.read_csv(datapath+fname_action2, encoding='gbk')
        #删除重复记录,只保留cate=8的数据
        action2=action2.drop_duplicates(keep='first')
        action2=action2[action2['cate']==8]
        action3=pd.read_csv(datapath+fname_action3, encoding='gbk')
        action3=action3.drop_duplicates(keep='first')
        action3=action3[action3['cate']==8]
        action4=pd.read_csv(datapath+fname_action4, encoding='gbk')
        action4=action4.drop_duplicates(keep='first')
        action4=action4[action4['cate']==8]
        sum_action1 = action2.append(action3, ignore_index=True)
        sum_action2=sum_action1.append(action4,ignore_index=True)
        #print(sum_action2.shape)
        #print(sum_action2.columns)
        #删除没有行为记录的购买
        only_buy_noaction_index = []
        group=sum_action2.groupby(['user_id','sku_id'])
        for key,value in group.groups.items():
            if len(value)==1:
                if sum_action2.iloc[value[0],4]==4:
                    only_buy_noaction_index+=list(value)
        sum_action2.drop(sum_action2.index[only_buy_noaction_index])
        sum_action2['user_id'] = sum_action2['user_id'].astype(int)
        all_action = sum_action2.sort_values(by=["user_id", "time", "sku_id"])
        all_action_cate8=all_action.loc[all_action['cate']==8]
        pickle.dump(all_action_cate8,open(interimvar+'all_action_cate8.pkl','wb'))
        pickle.dump(all_action, open(path, 'wb'))
    return all_action

#某段时间内用户的行为数据
def get_actions(start_date, end_date):
    path = interimvar+'all_action_cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(path):
        actions = pickle.load(open(path,'rb'))
    else:
        actions = gather_action()
        actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
        pickle.dump(actions, open(path, 'wb'))
    return actions

#某段时间商品评价情况
def get_comments_feature(start_date, end_date):
    path = interimvar+'comments_accumulate_cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(path):
        comments = pickle.load(open(path,'rb'))
    else:
        comments = pd.read_csv(datapath+fname_comment, encoding='gbk')
        comment_date_end = end_date
        comment_date_begin = comment_date[0]
        for date in reversed(comment_date):
            if date < comment_date_end:
                comment_date_begin = date
                break
        comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
        df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
        comments = pd.concat([comments, df], axis=1)
        comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
        comments=comments.set_index('sku_id')
        pickle.dump(comments, open(path, 'wb'))
    return comments


#用户某段时间总体特征
def extract_user_general_feature(start_date,end_date):
    path = interimvar+'user_general_feature_cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(path):
        actions = pickle.load(open(path, 'rb'))
    else:
        actions = None      #用户该时间段总特征
        section_actions=get_actions(start_date, end_date)#该时间段总行为数
        section_actions['time']=date_index(end_date)-section_actions['time'].map(date_index)
        #该段时间最近一次交互、购买时间
        first_inter_time=section_actions.loc[section_actions['type']!=4,['user_id','time']].sort_values(by=['user_id', 'time']).drop_duplicates('user_id',keep='first').rename(columns={'time':'first_inter_time'}).set_index('user_id')
        first_buy_time=section_actions.loc[section_actions['type']==4,['user_id','time']].sort_values(by=['user_id', 'time']).drop_duplicates('user_id',keep='first').rename(columns={'time':'first_buy_time'}).set_index('user_id')
        #该段时间最近3，7，10，15,30,45天有交互/购买的天数
        inter_days=None #交互天数
        buy_days=None   #发生购买天数
        for i in (3,7,10,15,30,45):
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            temp_actions = get_actions(start_days, end_date)
            temp_actions['time']=temp_actions['time'].map(get_day)
            if inter_days is None:
                inter_days=temp_actions[['user_id','time']].groupby('user_id')['time'].unique().apply(len).to_frame(name='before_3days_interdays')
                buy_days=temp_actions.loc[temp_actions['type']==4,['user_id','time']].groupby('user_id')['time'].unique().apply(len).to_frame(name='before_3days_buydays')
            else:
                inter_days=pd.merge(inter_days,temp_actions[['user_id','time']].groupby('user_id')['time'].unique().apply(len).to_frame(name='before_'+str(i)+'days_interdays'),left_index=True,right_index=True,how = 'outer')
                buy_days=pd.merge(buy_days,temp_actions.loc[temp_actions['type']==4,['user_id','time']].groupby('user_id')['time'].unique().apply(len).to_frame(name='before_'+str(i)+'days_buydays'),left_index=True,right_index=True,how = 'outer')
        #该段时间最近1, 2, 3, 4, 5, 7, 10, 15, 21, 30, 45天交互总次数（购买量）、交互过的不同品牌个数（购买品牌数）、转化率
        #buy_number=None                #购买量
        #buy_brand_number=None      #去重，购买品牌数
        inter_times= None   #交互总次数
        inter_brand_times=None#交互过的不同品牌数
        buy_ratio=None  #转化率
        for i in (1, 2, 3, 4, 5, 7, 10, 15, 21, 30, 45):
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            temp_actions = get_actions(start_days, end_date)
            temp_inter_brand_number=temp_actions[['user_id', 'sku_id', 'type']]
            ratio_type=pd.get_dummies(temp_inter_brand_number['type'],prefix='type')
            temp_ratio=pd.concat([temp_inter_brand_number[['user_id']],ratio_type],axis=1).groupby(['user_id'], as_index=False).sum()
            if inter_times is None:
                #buy_number=temp_actions.loc[temp_actions['type']==4,['user_id']].groupby('user_id',as_index=False).size().to_frame(name='before_1days_buynumbers') 
                #buy_brand_number=temp_actions.loc[temp_actions['type']==4,['user_id','sku_id']].drop_duplicates(keep='first').groupby(['user_id'],as_index=False).size().to_frame(name='before_1days_buybrandnumbers')
                inter_type=pd.get_dummies(temp_inter_brand_number['type'],prefix='before_1days_intertimes_type')
                inter_brand=pd.get_dummies(temp_inter_brand_number['type'],prefix='before_1days_interbrand_type')
                inter_times=pd.concat([temp_inter_brand_number[['user_id']],inter_type],axis=1).groupby(['user_id']).sum()
                temp_brand_times=pd.concat([temp_inter_brand_number[['user_id','sku_id']],inter_brand],axis=1).drop_duplicates(subset=['user_id','sku_id'],keep='first')
                del temp_brand_times['sku_id']
                inter_brand_times=temp_brand_times.groupby(['user_id']).sum()
                
                temp_ratio['before_1_days_user_skim_ratio']=temp_ratio['type_4'] / temp_ratio['type_1']
                temp_ratio['before_1_days_user_addcart_ratio']=temp_ratio['type_4'] / temp_ratio['type_2']
                temp_ratio['before_1_days_user_delcart_ratio']=temp_ratio['type_4'] / temp_ratio['type_3']
                temp_ratio['before_1_days_user_fav_ratio']=temp_ratio['type_4'] / temp_ratio['type_5']
                temp_ratio['before_1_days_user_click_ratio']=temp_ratio['type_4'] / temp_ratio['type_6']
                temp_ratio=temp_ratio.drop(['type_'+str(t) for t in range(1,7)],axis=1)
                temp_ratio=temp_ratio.set_index('user_id')
                buy_ratio = temp_ratio
            else:
               
                #buy_number = pd.merge(buy_number,temp_actions.loc[temp_actions['type']==4,['user_id']].groupby('user_id',as_index=False).size().to_frame(name='before_'+str(i)+'days_buynumbers'),left_index=True,right_index=True,how = 'outer')
                #buy_brand_number=pd.merge(buy_brand_number,temp_actions.loc[temp_actions['type']==4,['user_id','sku_id']].drop_duplicates(keep='first').groupby(['user_id'],as_index=False).size().to_frame(name='before_'+str(i)+'days_buybrandnumbers'))
                inter_type=pd.get_dummies(temp_inter_brand_number['type'],prefix='before_'+str(i)+'days_intertimes_type')
                inter_brand=pd.get_dummies(temp_inter_brand_number['type'],prefix='before_'+str(i)+'days_interbrand_type')
                inter_times=pd.merge(inter_times,pd.concat([temp_inter_brand_number[['user_id']],inter_type],axis=1).groupby(['user_id'], as_index=False).sum(),left_index=True,right_index=True,how = 'outer')
                temp_brand_times=pd.concat([temp_inter_brand_number[['user_id','sku_id']],inter_brand],axis=1).drop_duplicates(subset=['user_id','sku_id'],keep='first')
                del temp_brand_times['sku_id']
                inter_brand_times=pd.merge(inter_brand_times,temp_brand_times.groupby(['user_id']).sum(),left_index=True,right_index=True,how = 'outer')

                temp_ratio['before_'+str(i)+'_days_user_skim_ratio']=temp_ratio['type_4'] / temp_ratio['type_1']
                temp_ratio['before_'+str(i)+'_days_user_addcart_ratio']=temp_ratio['type_4'] / temp_ratio['type_2']
                temp_ratio['before_'+str(i)+'_days_user_delcart_ratio']=temp_ratio['type_4'] / temp_ratio['type_3']
                temp_ratio['before_'+str(i)+'_days_user_fav_ratio']=temp_ratio['type_4'] / temp_ratio['type_5']
                temp_ratio['before_'+str(i)+'_days_user_click_ratio']=temp_ratio['type_4'] / temp_ratio['type_6']
                temp_ratio=temp_ratio.drop(['type_'+str(t) for t in range(1,7)],axis=1)
                temp_ratio=temp_ratio.set_index('user_id')
                buy_ratio = pd.merge(buy_ratio,temp_ratio,left_index=True,right_index=True,how = 'outer')

        #合并用户特征
        actions = pd.merge(first_inter_time,first_buy_time,left_index=True,right_index=True,how='outer')
        actions = pd.merge(actions,inter_days,left_index=True,right_index=True,how ='outer')
        actions = pd.merge(actions,buy_days,left_index=True,right_index=True,how = 'outer')
        actions = pd.merge(actions,inter_times,left_index=True,right_index=True,how = 'outer')
        actions = pd.merge(actions,inter_brand_times,left_index=True,right_index=True,how = 'outer')
        actions = actions.fillna(0)
        pickle.dump(actions,open('temp_'+str(start_date)+'_'+str(end_date)+'.pkl','wb'))
        actions = get_dummies_for_pd(actions)
        actions = pd.merge(actions,buy_ratio,left_index=True,right_index=True,how = 'outer')
        actions = actions.fillna(0)
        pickle.dump(actions, open(path, 'wb'))
    return actions
    
# 某段时间商品总体特征
def extract_product_general_feature(start_date,end_date):
    path = interimvar+'product_general_cate8_feature_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(path):
        actions = pickle.load(open(path, 'rb'))
    else:
        actions = None		#商品该时间段总特征
        section_actions=get_actions(start_date, end_date)#该时间段总行为数
        section_actions['time']=date_index(end_date)-section_actions['time'].map(date_index)
        #该段时间第一次/最后一次商品交互/购买时间
        first_inter_time=section_actions.loc[section_actions['type']!=4,['sku_id','time']].sort_values(by=['sku_id', 'time']).drop_duplicates('sku_id',keep='last').rename(columns={'time':'first_inter_time'}).set_index('sku_id')
        late_inter_time=section_actions.loc[section_actions['type']!=4,['sku_id','time']].sort_values(by=['sku_id', 'time']).drop_duplicates('sku_id',keep='first').rename(columns={'time':'late_inter_time'}).set_index('sku_id')
        first_buy_time=section_actions.loc[section_actions['type']==4,['sku_id','time']].sort_values(by=['sku_id', 'time']).drop_duplicates('sku_id',keep='last').rename(columns={'time':'first_buy_time'}).set_index('sku_id')
        late_buy_time=section_actions.loc[section_actions['type']==4,['sku_id','time']].sort_values(by=['sku_id', 'time']).drop_duplicates('sku_id',keep='first').rename(columns={'time':'late_buy_time'}).set_index('sku_id')
        #返客数,不同天重复购买人数,并归一化
        temp_rebuy_user=section_actions.loc[section_actions['type']==4,['sku_id','user_id','time']].drop_duplicates(keep='first').groupby(['sku_id','user_id'],as_index=False).size().to_frame(name='diffbuydays')
        rebuy_user=temp_rebuy_user.loc[temp_rebuy_user['diffbuydays']>1,['sku_id','user_id']].groupby('sku_id').size().to_frame(name='rebuy_user_num')
        rebuy_user=norm_df(rebuy_user.loc[rebuy_user['rebuy_user_num']>1])
        #该段时间最近1, 2, 3, 4, 5, 7, 10, 15, 21, 30, 45天商品销量、转化率,并归一化
        buy_ratio=None  #转化率
        total_sales=None#周期销量
        for i in (1, 2, 3, 4, 5, 7, 10, 15, 21, 30, 45):
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            things=get_actions(start_days, end_date)
            temp_ratio = things[['sku_id','user_id','type']]
            total_sales=norm_df(things.loc[things['type']==4,['sku_id']].groupby('sku_id').size().to_frame(name='before_'+str(i)+'days_total_sales'))
            ratio_type = pd.get_dummies(temp_ratio['type'], prefix='type')
            if buy_ratio is None:

                temp_ratio = pd.concat([temp_ratio[['sku_id']], ratio_type], axis=1).groupby(['sku_id']).sum()
                temp_ratio['before_1_days_product_skim_ratio'] = temp_ratio['type_4'] / temp_ratio['type_1']
                temp_ratio['before_1_days_product_addcart_ratio'] = temp_ratio['type_4'] / temp_ratio['type_2']
                temp_ratio['before_1_days_product_delcart_ratio'] = temp_ratio['type_4'] / temp_ratio['type_3']
                temp_ratio['before_1_days_product_fav_ratio'] = temp_ratio['type_4'] / temp_ratio['type_5']
                temp_ratio['before_1_days_product_click_ratio'] =temp_ratio['type_4'] / temp_ratio['type_6']
                temp_ratio=temp_ratio.drop(['type_'+str(t) for t in range(1,7)],axis=1)
                buy_ratio=temp_ratio

            else:

                temp_ratio = pd.concat([temp_ratio[['sku_id']], ratio_type], axis=1).groupby(['sku_id']).sum()
                temp_ratio['before_'+str(i)+'_days_product_skim_ratio'] = temp_ratio['type_4'] / temp_ratio['type_1']
                temp_ratio['before_'+str(i)+'_days_product_addcart_ratio'] = temp_ratio['type_4'] / temp_ratio['type_2']
                temp_ratio['before_'+str(i)+'_days_product_delcart_ratio'] = temp_ratio['type_4'] / temp_ratio['type_3']
                temp_ratio['before_'+str(i)+'_days_product_fav_ratio'] = temp_ratio['type_4'] / temp_ratio['type_5']
                temp_ratio['before_'+str(i)+'_days_product_click_ratio'] =temp_ratio['type_4'] / temp_ratio['type_6']
                temp_ratio=temp_ratio.drop(['type_'+str(t) for t in range(1,7)],axis=1)
                buy_ratio=pd.merge(buy_ratio,temp_ratio,left_index=True,right_index=True,how = 'outer')

        #购买商品的总人数,归一化
        buy_product_total_user=norm_df(section_actions.loc[section_actions['type']==4,['sku_id','user_id']].drop_duplicates(keep='last').groupby('sku_id').size().to_frame(name='buy_product_total_user'))

        #最近几周，每周商品销量
        week_sales=None #一周每天销售数量累计
        for i in (7,14,21,28,35,42,49):
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            temp_week = get_actions(start_days, end_date)
            temp_week = temp_week.loc[temp_week['type']==4]
            temp_week['weekday'] = (pd.to_datetime(pd.Series(temp_week['time']))).apply(lambda x: x.weekday())
            del temp_week['time']
            if week_sales is None:
                weekday = pd.get_dummies(temp_week['weekday'], prefix='before_1_weekday')
                week_sales = pd.concat([temp_week['sku_id'], weekday], axis=1)
                week_sales = week_sales.groupby(['sku_id']).sum()
            else:
                weekday = pd.get_dummies(temp_week['weekday'], prefix='before_+'+str(i//7)+'_weekday')
                temp = pd.concat([temp_week['sku_id'], weekday], axis=1).groupby(['sku_id']).sum()
                week_sales=pd.merge(week_sales,temp,how='outer',left_index=True,right_index=True)
        week_sales=norm_df(week_sales)

        #合并所有商品特征
        actions = pd.merge(first_inter_time,late_inter_time,left_index=True,right_index=True,how='outer')
        actions = pd.merge(actions,first_buy_time,left_index=True,right_index=True,how ='outer')
        actions = pd.merge(actions,late_buy_time,left_index=True,right_index=True,how ='outer')
        actions = actions.fillna(0)
        actions = get_dummies_for_pd(actions)
        actions = pd.merge(actions,rebuy_user,left_index=True,right_index=True,how ='outer')
        actions = pd.merge(actions,buy_ratio,left_index=True,right_index=True,how ='outer')
        actions = pd.merge(actions,total_sales,left_index=True,right_index=True,how ='outer')
        actions = pd.merge(actions,buy_product_total_user,left_index=True,right_index=True,how ='outer')
        actions = pd.merge(actions,week_sales,left_index=True,right_index=True,how ='outer')
        actions = actions.fillna(0)
        pickle.dump(actions, open(path, 'wb'))
    return actions

# 某段时间 行为总体特征
def extract_action_general_feature(start_date, end_date):
    path = interimvar+'action_accumulate_cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(path):
        actions = pickle.load(open(path,'rb'))
    else:
        actions = None
        for i in (1, 2, 3, 4, 5, 7, 10, 15, 21, 30, 45):
            start_days = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=i)
            start_days = start_days.strftime('%Y-%m-%d')
            temp_action = get_actions(start_days, end_date)
            temp_action=temp_action[['user_id', 'sku_id', 'type']]
            df = pd.get_dummies(temp_action['type'], prefix=str(i)+'days_pair_type')
            temp=pd.concat([temp_action[['user_id', 'sku_id']], df], axis=1).groupby(['user_id', 'sku_id']).sum()
            if actions is None:
                actions = temp
            else:
                actions=pd.merge(actions,temp,how='outer',left_index=True,right_index=True)
        pickle.dump(actions, open(path, 'wb'))
    return actions


#某段时间下单的(user_id,sku_id),label
def accumulate_buy_action(start_date, end_date):
    path = interimvar+'label_user_sku_cate8_%s_%s.pkl' % (start_date, end_date)
    if os.path.exists(path):
        actions = pickle.load(open(path,'rb'))
    else:
        actions = get_actions(start_date, end_date)
        actions = actions[actions['type_4'] == 1]
        actions = actions.groupby(['user_id', 'sku_id'], as_index=False).sum()
        actions['label'] = 1
        actions = actions[['user_id', 'sku_id','label']]
        pickle.dump(actions, open(path, 'wb'))
    return actions
#测试集
def get_testset(train_start_date, train_end_date):
    path = interimvar+'testset_cate8_%s_%s.pkl' % (train_start_date, train_end_date)
    if os.path.exists(path):
        actions = pickle.load(open(path,'rb'))
    else:
        user = deal_user().set_index('user_id')
        product = deal_product().set_index('sku_id')
        user_cover = extract_user_general_feature(train_start_date, train_end_date)
        product_cover = extract_product_general_feature(train_start_date, train_end_date)
        comment = get_comments_feature(train_start_date, train_end_date)
        pair_acc=extract_action_general_feature(train_start_date, train_end_date)


        actions = pd.merge(pair_acc, user, how='left',left_index=True,right_on=True)
        actions = pd.merge(actions, user_cover, how='left', left_index=True,right_index=True)
        actions = pd.merge(actions, product, how='left', left_index=True,right_index=True)
        actions = pd.merge(actions, product_cover, how='left', left_index=True,right_index=True)
        actions = pd.merge(actions, comment, how='left', left_index=True,right_index=True)

        actions = actions.fillna(0)
        pickle.dump(actions, open(path, 'wb'))

    users = actions.index
    actions = pd.DataFrame(actions.values,columns=actions.columns)
    return users, actions
#训练集
def get_trainingset(train_start_date, train_end_date, test_start_date, test_end_date):
    path = interimvar+'trainingset_cate8_%s_%s_%s_%s.pkl' % (train_start_date, train_end_date, test_start_date, test_end_date)
    if os.path.exists(path):
        actions = pickle.load(open(path,'rb'))
    else:
        user = deal_user().set_index('user_id')
        product = deal_product().set_index('sku_id')
        user_cover = extract_user_general_feature(train_start_date, train_end_date)
        product_cover = extract_product_general_feature(train_start_date, train_end_date)
        comment = get_comments_feature(train_start_date, train_end_date)
        pair_acc=extract_action_general_feature(train_start_date, train_end_date)
        labels = accumulate_buy_action(test_start_date, test_end_date).set_index(['user_id','sku_id'])

        actions = pd.merge(pair_acc, user, how='left',left_index=True,right_on=True)
        actions = pd.merge(actions, user_cover, how='left', left_index=True,right_index=True)
        actions = pd.merge(actions, product, how='left', left_index=True,right_index=True)
        actions = pd.merge(actions, product_cover, how='left', left_index=True,right_index=True)
        actions = pd.merge(actions, comment, how='left', left_index=True,right_index=True)
        actions = pd.merge(actions, labels, how='left', left_index=True,right_index=True)
        actions = actions.fillna(0)
        pickle.dump(actions, open(path, 'wb'))

    labels = actions['label'].copy()
    users = actions.index
    actions = pd.DataFrame(actions.values,columns=actions.columns)

    return users, actions, labels

def evalution(pred, label):

    actions = label
    result = pred

    # 所有用户商品对
    all_user_item_pair = actions['user_id'].map(str) + '-' + actions['sku_id'].map(str)
    all_user_item_pair = np.array(all_user_item_pair)
    # 所有购买用户
    all_user_set = actions['user_id'].unique()

    # 所有品类中预测购买的用户
    all_user_test_set = result['user_id'].unique()
    all_user_test_item_pair = result['user_id'].map(str) + '-' + result['sku_id'].map(str)
    all_user_test_item_pair = np.array(all_user_test_item_pair)

    # 计算所有用户购买评价指标
    pos, neg = 0,0
    for user_id in all_user_test_set:
        if user_id in all_user_set:
            pos += 1
        else:
            neg += 1
    all_user_acc = 1.0 * pos / ( pos + neg)
    all_user_recall = 1.0 * pos / len(all_user_set)
    print('所有用户中预测购买用户的准确率为 ' + str(all_user_acc))
    print('所有用户中预测购买用户的召回率' + str(all_user_recall))

    pos, neg = 0, 0
    for user_item_pair in all_user_test_item_pair:
        if user_item_pair in all_user_item_pair:
            pos += 1
        else:
            neg += 1
    all_item_acc = 1.0 * pos / ( pos + neg)
    all_item_recall = 1.0 * pos / len(all_user_item_pair)
    print('所有用户中预测购买商品的准确率为 ' + str(all_item_acc))
    print('所有用户中预测购买商品的召回率' + str(all_item_recall))
    F11 = 6.0 * all_user_recall * all_user_acc / (5.0 * all_user_recall + all_user_acc)
    F12 = 5.0 * all_item_acc * all_item_recall / (2.0 * all_item_recall + 3 * all_item_acc)
    score = 0.4 * F11 + 0.6 * F12
    print('F11=' + str(F11))
    print('F12=' + str(F12))
    print('score=' + str(score))

all_action=get_trainingset('2016-02-01','2016-04-11','2016-04-11','2016-04-16')
sub_action=get_testset('2016-02-01','2016-04-16')
