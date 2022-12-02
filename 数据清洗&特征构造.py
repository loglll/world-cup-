import pandas as pd


def sum_dict(dict1, dict2):
    temp = dict()
    for key in dict1.keys() | dict2.keys():
        temp[key] = sum([d.get(key, 0) for d in (dict1, dict2)])
    return temp


if __name__ == '__main__':
    df = pd.read_csv('results.csv')
    df['date'] = pd.to_datetime(df['date'])
    # 查看导入的数据
    # print(df.info())
    # tournament字段记录了赛事名称，那么我们这里取世界上比较有名的国际性赛事及,且为了样本足够大，其资格赛也取上
    target_tournament = [
        'UEFA Nations League', 'UEFA Euro qualification', 'UEFA Euro', 'Oceania Nations Cup qualification',
        'Oceania Nations Cup', 'Gold Cup qualification', 'Gold Cup', 'FIFA World Cup qualification',
        'FIFA World Cup', 'Copa América', 'CONIFA European Football Cup', 'CONIFA Africa Football Cup',
        'Confederations Cup', 'CONCACAF Nations League qualification', 'CONCACAF Nations League',
        'African Cup of Nations qualification', 'African Cup of Nations', 'AFC Asian Cup qualification', 'AFC Asian Cup'
    ]
    df_target = df.loc[df.tournament.isin(target_tournament)]
    # 先验证预测2018俄罗斯世界杯，此处先将2018的世界杯数据除去（2018-6-14）
    df_target = df_target.loc[df_target.date < '2018-6-14']
    # 将无关特征删去，此处仅留下主客队、主客队进球数,日期留下方便后续
    df_target = df_target.drop(['tournament', 'city', 'country', 'neutral'], axis=1)
    # 查看剩下的四个特征，共有17749条数据，均无缺失值，主客队进球数均为浮点型，没有问题
    # print(df_target.info())
    # 利用现有特征扩充更多的特征：每场比赛的结果(result)、主队参赛次数(home_times)、客队参赛次数(away_times)、主队胜利次数(home_win)、客队胜利次数(away_win)、主队胜率(home_rate_of_win)、客队胜率(away_rate_of_win)、主队总进球数(home_goal)、客队总进球数(away_goal)、主队场均进球(home_avg_goal)、客队场均进球(away_avg_goal)
    # 比赛结果(result):主队胜置为1，客队胜置为2，平局置为0
    df_target['result'] = df_target.apply(lambda x: 1 if x['home_score'] > x['away_score'] else 2 if x['home_score'] < x['away_score'] else 0,axis=1)
    # 主队参赛次数(home_times)、客队参赛次数(away_times):将主客队累计参赛数传入字典，并通过apply()获取字典key值对应的value
    home_times_dict = df_target['home_team'].value_counts().to_dict()
    away_times_dict = df_target['away_team'].value_counts().to_dict()
    sum_times_dict = sum_dict(home_times_dict,away_times_dict)
    df_target['home_times'] = df_target['home_team'].apply(lambda x: sum_times_dict[x])
    df_target['away_times'] = df_target['away_team'].apply(lambda x: sum_times_dict[x])
    # 主队胜利次数(home_win)、客队胜利次数(away_win):
    home_win_dict = df_target.loc[df_target.result == 1,'home_team'].value_counts().to_dict()
    away_win_dict = df_target.loc[df_target.result == 2,'away_team'].value_counts().to_dict()
    sum_win_dict = sum_dict(home_win_dict,away_win_dict)
    df_target['home_win'] = df_target.apply(lambda x: sum_win_dict[x['home_team']] if x['home_team'] in sum_win_dict.keys() else 0,axis=1)
    df_target['away_win'] = df_target.apply(lambda x: sum_win_dict[x['away_team']] if x['away_team'] in sum_win_dict.keys() else 0,axis=1)
    # 主队胜率(home_rate_of_win)、客队胜率(away_rate_of_win)
    df_target['home_rate_of_win'] = df_target.apply(lambda x: x['home_win'] / x['home_times'],axis=1)
    df_target['away_rate_of_win'] = df_target.apply(lambda x: x['away_win'] / x['away_times'],axis=1)
    # 主队总进球数(home_goal)、客队总进球数(away_goal)、主队场均进球(home_avg_goal)、客队场均进球(away_avg_goal):先算总进球数，再算场均进球数
    home_goal_dict = df_target[['home_team','home_score']].groupby('home_team').sum().to_dict()['home_score']
    away_goal_dict = df_target[['away_team', 'away_score']].groupby('away_team').sum().to_dict()['away_score']
    sum_goal_dict = sum_dict(home_goal_dict,away_goal_dict)
    df_target['home_goal'] = df_target['home_team'].apply(lambda x: sum_goal_dict[x])
    df_target['away_goal'] = df_target['away_team'].apply(lambda x: sum_goal_dict[x])
    df_target['home_avg_goal'] = df_target.apply(lambda x: x['home_goal'] / x['home_times'],axis=1)
    df_target['away_avg_goal'] = df_target.apply(lambda x: x['away_goal'] / x['away_times'], axis=1)

    # 特征扩充完毕，所有数据都存放在名为df_target的DateFrame中，下面进行数据预处理
    # 先将DateFrame各个特征取出来转化为对应的Series,并存放到相对于的变量中
    # print(df_target.info())
    # date = df_target['date']
    # home_team = df_target['home_team']
    # away_team = df_target['away_team']
    # home_score = df_target['home_score']
    # away_score = df_target['away_score']
    # result = df_target['result']
    # home_times = df_target['home_times']
    # away_times = df_target['away_times']
    # home_win = df_target['home_win']
    # away_win = df_target['away_win']
    # home_rate_of_win = df_target['home_rate_of_win']
    # away_rate_of_win = df_target['away_rate_of_win']
    # home_goal = df_target['home_goal']
    # away_goal = df_target['away_goal']
    # home_avg_goal = df_target['home_avg_goal']
    # away_avg_goal = df_target['away_avg_goal']

    # 数据预处理，除日期主客队名称及结果外，其余特征均采用标准分数的方法标准化
    '''
    标准分数（z-score）是一个分数与平均数的差再除以标准差的过程。
    用公式表示为：z=(x-μ)/σ。
    其中x为某一具体分数，μ为平均数，σ为标准差
    '''
    df_feature = df_target.drop(['date','home_team','away_team','result'],axis=1)
    df_normalizing = (df_feature - df_feature.mean()) / (df_feature.std())
    # df_normalizing.to_csv('test3.csv',encoding='utf-8',index=False)
    # 标准化后的数据与主客队名称及结果(result)连接起来,作为机器学习的数据集
    data_set = pd.concat([df_target[['home_team', 'away_team']],df_normalizing,df_target['result']],axis=1)
    data_set = data_set.reset_index(drop=True)
    # 至此，数据清洗及特征工程基本完成，保存到csv中
    # data_set.to_csv('data_set_with_team_name.csv',encoding='utf-8',index=False)




