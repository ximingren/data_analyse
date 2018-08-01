# -*- coding: UTF-8 -*-
import pandas as pd
import os


def split(data, result_col, need_split_col):
    """
    拆分数据
    :param data: 预处理过的总数据
    :param result_col: 拆分结果集的列索引
    :param need_split_col:拆分依据的列
    :return: split_result:拆分结果集
    """
    unitPrice = list()  # 存放单价的值
    count = list()  # 存储每一组需要拆分成多少小组的值
    global split_value  # 使其变为全局变量，能够在另外一个函数中用到
    split_result = pd.DataFrame(columns=result_col)  # 创建个空的DataFrame，以存放拆分后的结果
    # 对数据的每一行进行判断，若数据大于阈值则需要被分割
    for i in range(len(need_split_col)):
        eachData = list(data.iloc[i].name)  # 因为前面分组过了，所以这里输出来的是分组层次索引的数值
        # 加上相应列的值,先取出第i行,然后取出相应的列
        eachData.append(data.iloc[i]['客户'])
        eachData.append(data.iloc[i]['利润中心'])
        eachData.append(data.iloc[i]['产品号'])
        eachData.append(data.iloc[i]['税率'])
        param_len = len(eachData)  # 固定参数长度
        # 如果数值为0，则后续的不用计算具体的值
        if data.iloc[i]['销售数量'] == 0:
            # 单价为0
            unitPrice.append(0)
            # 只用加一行
            count.append(1)
            # 加上相应的列
            eachData.append(data.iloc[i]['主营业务收入'])
            eachData.append(data.iloc[i]['销项税额'])
            eachData.append(data.iloc[i]['含税销售额(净额)'])
        # 如果数值不为0，则计算相应的值
        else:
            unitPrice.append(data.iloc[i]['主营业务收入'] / data.iloc[i]['销售数量'])  # 计算每件物品的单价,主营业务收入/销售数量=单价
            split_value = need_split_col.iloc[i]  # split_value是最后的拆分值
            # 判断是否大于阈值
            if split_value > split_threshold:
                #  若一直大于则一直二分下去
                while (split_value > split_threshold):
                    split_value = split_value / 2.0
            count.append(int(data.iloc[i]['主营业务收入'] / split_value))  # 计算要分成多少组才能小于阈值,总值/划分结果值=组数
            sale_num = int(split_value / unitPrice[i])  # 计算出销售数量,销售数量=拆分值/单价
            afterTax = float('%.2f' % (split_value * data.iloc[i]['税率']))  # 计算出销项税额,主营业务收入*税率=销项税额
            eachData.append(sale_num)  # 添加销售数量
            eachData.append(float('%.2f' % (split_value)))  # 添加主营业务收入，保留2位小数
            eachData.append(afterTax)  # 添加销项税额
            eachData.append(float('%.2f' % (split_value + afterTax)))  # 主营业务收入+销项税额=含税销售额(净额)，并天津爱
        eachData = pd.Series(eachData, index=result_col)  # 拆分值的每一行
        # 第i个客户的数据经过了count[i]组分割后的值小于阈值，则添加count[i]组
        split_result = integrate(data, eachData, split_value, split_result, param_len, unitPrice, count[i], i)
    # 返回经过所有行的拆分后的拆分结果集
    return split_result


def integrate(data, eachData, split_value, split_result, param_len, unitPrice, num, i):
    """
    将拆分行一行一行得整合出来
    :param data:  预处理过得总数据
    :param eachData: 拆分过的行数据
    :param split_value: 最后的拆分值
    :param split_result: 拆分结果集
    :param param_len: 固定参数长度
    :param num: 分组数
    :param i: 第i个用户，即是第i行
    :return: 拆分结果集
    """
    income_mistakes = 0  # 主营业务收入误差,保留2位小数后产生的误差
    for k in range(num):
        """计算保留2位小数后和原始数据的误差,累加保留2位小数后和划分结果值的差值
           eachData[param_len]是销售数量,所以each[param_len+1]是主营业务收入
           累加每一组的保留小数后的主营业务收入和保留前的误差
        """
        income_mistakes = income_mistakes + eachData[param_len + 1] - split_value
        # 到了最后一列要根据原始数据和划分后的数据比较，更改数据
        if k == num - 1:
            # 要销售数量全部加起来等于原数据的值,总销售数量-总的划分销售数量=最后一行的销售数量
            eachData[param_len] = int(data.iloc[i]['销售数量'] - eachData[param_len] * (num - 1))
            # 销售数量乘于单价，再加上差值,即为最后一行的主营业务收入
            eachData[param_len + 1] = float('%.2f' % (eachData[param_len] * unitPrice[i] + income_mistakes))
            # 同时也要改变销项税额，主营业务收入乘于税率
            eachData[param_len + 2] = float('%.2f' % (eachData[param_len + 1] * data.iloc[i]['税率']))
            # 同时也有含税销售额(净额)
            eachData[param_len + 3] = float('%.2f' % (eachData[param_len + 2] + eachData[param_len + 1]))
        # DataFrame添加后不改变原来的DataFrame，返回的是添加后的dataframe
        split_result = split_result.append(eachData, ignore_index=True)
    # 返回添加一轮后的拆分结果
    return split_result


def diff(split_result):
    """
    计算出差异值
    :param split_result: 拆分后的结果集
    :return: 添加差异一列的结果集
    """
    split_result['销项税额(计算)'] = split_result.apply(lambda x,: x['主营业务收入'] * x['税率'], axis=1)  # 计算出来的销项税额
    split_result['差异'] = split_result.apply(lambda x: x['销项税额(计算)'] - x['销项税额'], axis=1)  # 计算出来的销项税额和未计算的销项税额的差值
    return split_result


def create_directory(path):
    """
    创建新目录(如果不存在)
    :param path:
    :return:
    """
    try:
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
    except Exception as e:
        print(e)
    finally:
        pass


def pre_process(raw_data, reduce_data, reference_data, need_col, need_reference_col, result_col,group_index):
    """
    对总数据表进行预处理
    步骤有：1.依据参照表对总数据表和核减表进行合并添加相应的列
           2.依据核减表对总数据表进行核减操作
           3.对总数据表进行补充处理操作
    :param raw_data:  总数据表
    :param reduce_data: 核减表
    :param reference_data: 参照表
    :param need_col: 需要保留的列
    :param need_reference_col: 参照表中需要保留的列
    :param group_index: 分组依据列
    :return: 预处理后的总数据表
    """
    raw_data = raw_data[need_col].fillna(0)  # 并将nan值填充0
    reduce_data = reduce_data[need_col].fillna(0)  # 并将nan值填充0
    #  合并两个表,即是将raw数据表和reduce数据表添加购货单位和利润中心两列。依据客户这一列进行合并
    merged_raw_data = raw_data.merge(reference_data[need_reference_col], left_on='客户', right_on='客户', how='left')
    merged_reduce_data = reduce_data.merge(reference_data[need_reference_col], left_on='客户', right_on='客户',
                                           how='left').dropna()
    summary_data = merged_raw_data.groupby(group_index).agg(
        {'销售数量': 'sum', '主营业务收入': 'sum', '销项税额': 'sum', '含税销售额(净额)': 'sum'})  # 分组操作并聚合运算
    reduce_data = merged_reduce_data.groupby(group_index).agg(
        {'销售数量': 'sum', '主营业务收入': 'sum', '销项税额': 'sum', '含税销售额(净额)': 'sum'})  # 分组操作并聚合运算
    data = (summary_data - reduce_data)  # 核减操作
    data = data.combine_first(summary_data)  # 对未对应得上的数据进行填充处理
    merged_raw_data = merged_raw_data.reindex(columns=result_col).set_index(['购货单位', '产品'])  # 对未分组的数据进行处理，设置列排序，设置层次索引
    data = data.reindex(columns=result_col.drop(['购货单位', '产品']))  # 设置列排序（其实是新添多个列）
    group_index2 = ['购货单位', '产品', '客户', '利润中心', '产品号', '税率']
    merged_raw_data=merged_raw_data.groupby(group_index2).sum().reset_index().set_index(group_index)
    data = data.combine_first(merged_raw_data)  # 两个表填充
    data.to_excel('./result/before_split_total_data.xls')  # 导出拆分之前的总数据，但是核减过后的
    return data


if __name__ == '__main__':
    """下面的配置变量有点多,找机会改一下看怎么弄，怎么简化它"""
    pd.set_option('display.width', 200)
    reduce_file = '8.xls'
    rawData_file = "7.xls"
    reference_file = '参照表 (1).xlsx'
    create_directory("./result")
    split_result_file = './result/final_result5.xls'
    # 设置阈值
    split_threshold = 30000
    # 经过分组整合后的数据
    reduce_data = pd.read_excel(reduce_file)  # 核减的数值
    raw_data = pd.read_excel(rawData_file)  # 为处理过的总数据
    reference_data = pd.read_excel(reference_file).fillna(0)  # 参照表的数据,并将nan值填充0
    need_col = pd.Index(['客户', '产品号', '产品', '税率', '销售数量', '主营业务收入', '销项税额', '含税销售额(净额)'])  # 处理raw数据需要用到的列
    need_reference_col = pd.Index(['客户', '购货单位', '利润中心'])  # 参照表需要用到的列
    result_col = pd.Index(['购货单位', '产品', '客户', '利润中心', '产品号', '税率', '销售数量', '主营业务收入', '销项税额', '含税销售额(净额)'])  # 结果集的列
    group_index = ['购货单位', '产品']  # 分组依据的索引
    print("进行数据预处理")
    data = pre_process(raw_data, reduce_data, reference_data, need_col, need_reference_col, result_col,group_index)  # 预处理操作
    # 取得是‘主营业务收入’,对这一列数据进行判断
    need_split_col = data['主营业务收入']
    # 拆分数据,返回的是拆分后的结果
    print("进行数据拆分")
    split_result = split(data, result_col, need_split_col)
    # 计算出拆分前与拆分后的差异值
    print("进行计算差异")
    diff_result = diff(split_result)
    diff_result=diff_result.set_index(group_index)
    diff_result.to_excel(split_result_file)
