import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

data_set = pd.read_csv('data_set.csv')
data_set = data_set.loc[data_set.result != 0]  # 此处为求算法精度后，将平局去掉


# 调用train_test_split()方法，划分训练集和测试集,其中home_score、away_score不作为特征训练
X = data_set.drop(['home_goal','away_goal','result','home_score','away_score'],axis=1)  # 模型改进3中，减少特征home_goal及away_goal准确率会稍微高一点
y = data_set['result']
# X_train为训练特征，y_train为训练目标，X_test为测试特征，y_test为测试目标，test_size为测试集的占比，此处取30%;random_state为随机数种子，只要不为0或者空即可
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=7)


# 调用逻辑回归算法
print("=====逻辑回归算法=====")
logreg = LogisticRegression(C=1,solver='liblinear',multi_class ='auto')
logreg.fit(X_train, y_train)
score_log = logreg.score(X_train, y_train)
score2_log = logreg.score(X_test, y_test)
y_log_pred = logreg.predict(X_test)
print("训练集精度: ", '%.3f' % score_log)
print("测试集精度: ", '%.3f' % score2_log)
print("平均绝对误差: ",mean_absolute_error(y_test,y_log_pred))
print('Precision: %.3f' % precision_score(y_test, y_log_pred))
print('Recall: %.3f' % recall_score(y_test, y_log_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_log_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_log_pred))

# 调用SVM支持向量机算法
print("=====SVM支持向量机=====")
clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf.fit(X_train, y_train)
score_svm = clf.score(X_train, y_train)
score2_svm = clf.score(X_test, y_test)
y_svm_pred = clf.predict(X_test)
print("训练集精度: ", '%.3f' % score_svm)
print("测试集精度: ", '%.3f' % score2_svm)
print("平均绝对误差: ",mean_absolute_error(y_test,y_svm_pred))
print('Precision: %.3f' % precision_score(y_test, y_svm_pred))
print('Recall: %.3f' % recall_score(y_test, y_svm_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_svm_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_svm_pred))

# 调用随机森林算法
print("=====随机森林算法=====")
rf = RandomForestClassifier(max_depth=20,n_estimators=1000,random_state=0)
rf.fit(X_train, y_train)
score_rf = rf.score(X_train, y_train)
score2_rf = rf.score(X_test, y_test)
y_rf_pred = rf.predict(X_test)
print("训练集精度: ", '%.3f' % score_rf)
print("测试集精度: ", '%.3f' % score2_rf)
print("平均绝对误差: ",mean_absolute_error(y_test,y_rf_pred))
print('Precision: %.3f' % precision_score(y_test, y_rf_pred))
print('Recall: %.3f' % recall_score(y_test, y_rf_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_rf_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_rf_pred))

# 调用神经网络算法
print("=====神经网络算法=====")
mlp = MLPClassifier(hidden_layer_sizes=10,max_iter=1000)
mlp.fit(X_train, y_train)
score_mlp = mlp.score(X_train, y_train)
score2_mlp = mlp.score(X_test, y_test)
y_mlp_pred = mlp.predict(X_test)
print("训练集精度: ", '%.3f' % score_mlp)
print("测试集精度: ", '%.3f' % score2_mlp)
print("平均绝对误差: ",mean_absolute_error(y_test,y_mlp_pred))
print('Precision: %.3f' % precision_score(y_test, y_mlp_pred))
print('Recall: %.3f' % recall_score(y_test, y_mlp_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_mlp_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_mlp_pred))

# 调用决策树算法
print("=====决策树算法=====")
tree=DecisionTreeClassifier(max_depth=50,random_state=0)
tree.fit(X_train, y_train)
score_tree = tree.score(X_train, y_train)
score2_tree = tree.score(X_test, y_test)
y_tree_pred = tree.predict(X_test)
print("训练集精度: ", '%.3f' % score_tree)
print("测试集精度: ", '%.3f' % score2_tree)
print("平均绝对误差: ",mean_absolute_error(y_test,y_tree_pred))
print('Precision: %.3f' % precision_score(y_test, y_tree_pred))
print('Recall: %.3f' % recall_score(y_test, y_tree_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_tree_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_tree_pred))



# 导入2018年世界杯的对阵图
world_cup_2018 = pd.read_csv('2018_worldcup.csv')
# 只取小组赛的对阵
df_group_match_2018 = world_cup_2018.loc[world_cup_2018.Stage.str.contains('Group')]
# 只保留主客队名，其余去掉
df_group_match_2018_col = df_group_match_2018.columns.tolist()
df_group_match_2018_col.remove('主队')
df_group_match_2018_col.remove('客队')
df_group_match_2018 = df_group_match_2018.drop(df_group_match_2018_col,axis=1)
# 世界杯赛场上没有主客之分，这里简单用世界排名确定主客，排名靠前的的为主，故导入排名数据
ranking = pd.read_csv('Current FIFA rank-2018.csv')
# 在df_group_match_2018中插入两个新列
df_group_match_2018.insert(1,'rank_1',df_group_match_2018['主队'].map(ranking.set_index('Team')['Current FIFA rank']))
df_group_match_2018.insert(3,'rank_2',df_group_match_2018['客队'].map(ranking.set_index('Team')['Current FIFA rank']))
# 根据排名高低，修正主客队
a_list = []
for index,row in df_group_match_2018.iterrows():
    if row['rank_1'] < row['rank_2']:
        a_list.append({'主队':row['主队'],'客队':row['客队']})
    else:
        a_list.append({'主队':row['客队'],'客队':row['主队']})

df_group_match_2018 = pd.DataFrame(a_list)

# 从data_set_with_team_name数据集中匹配各队的特征
df_info = pd.read_csv('data_set_with_team_name.csv')
home_info_dict = df_info[['home_team','home_times','home_win','home_rate_of_win','home_goal','home_avg_goal']].groupby('home_team').mean().to_dict()
away_info_dict = df_info[['away_team','away_times','away_win','away_rate_of_win','away_goal','away_avg_goal']].groupby('away_team').mean().to_dict()
df_group_match_2018['home_times'] = df_group_match_2018.apply(lambda x: home_info_dict['home_times'][x['主队']],axis=1)
df_group_match_2018['away_times'] = df_group_match_2018.apply(lambda x: away_info_dict['away_times'][x['客队']],axis=1)
df_group_match_2018['home_win'] = df_group_match_2018.apply(lambda x: home_info_dict['home_win'][x['主队']],axis=1)
df_group_match_2018['away_win'] = df_group_match_2018.apply(lambda x: away_info_dict['away_win'][x['客队']],axis=1)
df_group_match_2018['home_rate_of_win'] = df_group_match_2018.apply(lambda x: home_info_dict['home_rate_of_win'][x['主队']],axis=1)
df_group_match_2018['away_rate_of_win'] = df_group_match_2018.apply(lambda x: away_info_dict['away_rate_of_win'][x['客队']],axis=1)
# df_group_match_2018['home_goal'] = df_group_match_2018.apply(lambda x: home_info_dict['home_goal'][x['主队']],axis=1)
# df_group_match_2018['away_goal'] = df_group_match_2018.apply(lambda x: away_info_dict['away_goal'][x['客队']],axis=1)
df_group_match_2018['home_avg_goal'] = df_group_match_2018.apply(lambda x: home_info_dict['home_avg_goal'][x['主队']],axis=1)
df_group_match_2018['away_avg_goal'] = df_group_match_2018.apply(lambda x: away_info_dict['away_avg_goal'][x['客队']],axis=1)
df_group_match_2018['result'] = None


# 用逻辑回归预测
output_info = df_group_match_2018
pred_set = df_group_match_2018.drop(['主队','客队','result'],axis=1)
predictions = logreg.predict(pred_set)


# for i in range(48):
#     print(output_info.iloc[i,0] + '  Vs.  ' + output_info.iloc[i,1])
#     if predictions[i] == 1:
#         print('Winner:' + output_info.iloc[i,0])
#     else:
#         print('Winner:' + output_info.iloc[i,1])
#
#     print(output_info.iloc[i,0] + '--胜出的可能性为:' + '%.3f'%(logreg.predict_proba(pred_set)[i][0]))
#     print(output_info.iloc[i, 1] + '--胜出的可能性为:' + '%.3f' % (logreg.predict_proba(pred_set)[i][1]))
#     print('')
#     print('=============')
#     print('')


# 预测2022年卡塔尔世界杯
# 读入2022年小组赛对阵的球队
df_group_match_2022 = pd.read_csv('2022_World_Cup.csv')
# 从data_set_with_team_name数据集中匹配各队的特征
df_info = pd.read_csv('data_set_with_team_name.csv')
home_info_dict = df_info[['home_team','home_times','home_win','home_rate_of_win','home_goal','home_avg_goal']].groupby('home_team').mean().to_dict()
away_info_dict = df_info[['away_team','away_times','away_win','away_rate_of_win','away_goal','away_avg_goal']].groupby('away_team').mean().to_dict()
df_group_match_2022['home_times'] = df_group_match_2022.apply(lambda x: home_info_dict['home_times'][x['home_team']],axis=1)
df_group_match_2022['away_times'] = df_group_match_2022.apply(lambda x: away_info_dict['away_times'][x['away_team']],axis=1)
df_group_match_2022['home_win'] = df_group_match_2022.apply(lambda x: home_info_dict['home_win'][x['home_team']],axis=1)
df_group_match_2022['away_win'] = df_group_match_2022.apply(lambda x: away_info_dict['away_win'][x['away_team']],axis=1)
df_group_match_2022['home_rate_of_win'] = df_group_match_2022.apply(lambda x: home_info_dict['home_rate_of_win'][x['home_team']],axis=1)
df_group_match_2022['away_rate_of_win'] = df_group_match_2022.apply(lambda x: away_info_dict['away_rate_of_win'][x['away_team']],axis=1)
# df_group_match_2022['home_goal'] = df_group_match_2022.apply(lambda x: home_info_dict['home_goal'][x['home_team']],axis=1)
# df_group_match_2022['away_goal'] = df_group_match_2022.apply(lambda x: away_info_dict['away_goal'][x['away_team']],axis=1)
df_group_match_2022['home_avg_goal'] = df_group_match_2022.apply(lambda x: home_info_dict['home_avg_goal'][x['home_team']],axis=1)
df_group_match_2022['away_avg_goal'] = df_group_match_2022.apply(lambda x: away_info_dict['away_avg_goal'][x['away_team']],axis=1)
df_group_match_2022['result'] = None


# 用逻辑回归预测2022世界杯小组赛
output_info = df_group_match_2022
# print(output_info.info())
pred_set = df_group_match_2022.drop(['match_index','主队','客队','home_team','away_team','result'],axis=1)
predictions = logreg.predict(pred_set)

for i in range(48):
    print('====*====' + output_info.iloc[i, 0] + '====*====')
    print(output_info.iloc[i, 1] + '  Vs.  ' + output_info.iloc[i, 2])
    if predictions[i] == 1:
        print('Winner:' + output_info.iloc[i, 1])
    else:
        print('Winner:' + output_info.iloc[i, 2])
    print(output_info.iloc[i, 1] + '--胜出的可能性为:' + '%.3f' % (logreg.predict_proba(pred_set)[i][0]))
    print(output_info.iloc[i, 2] + '--胜出的可能性为:' + '%.3f' % (logreg.predict_proba(pred_set)[i][1]))
    print('===========*============')
    print('')
'''

