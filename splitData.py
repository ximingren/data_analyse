# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np

"""
    对原始数据进行处理,下面这个函数将原始数据
"""


def aggergate(file):
    # 读取数据
    rawData = pd.DataFrame(pd.read_excel(file))
    # 对这几列进行拆分运算
    sumResult = rawData[need_col].groupby(['客户名称', '产品号', '客户', '税率']).agg('sum')
    # # 汇总表输出
    sumResult.to_excel('./split/gather.xls')
    return sumResult


"""
    拆分数据
"""


def split(data, split_result, need_split, num=0):
    global count
    # 对数据的每一行进行判断，若数据大于阈值则需要被分割
    for i in range(len(need_split)):
        # 计算每件物品的单价
        unitPrice.append(data['主营业务收入'][i] / data['销售数量'][i])
        # 判断是否大于阈值,split_value是最后的拆分值
        split_value = need_split[i]
        if split_value > value:
            #  若一直大于则一直二分下去
            while (split_value > value):
                # 计算二分了多少次
                num = num + 1
                split_value = split_value / 2.0
        # 计算要分成多少组才能小于阈值
        count.append(int(data['主营业务收入'][i] / split_value))
        need_split[i] = split_value
        # 第i行的名称，因为前面整合的时候分组了，所以这里输出来的是分组的数值
        eachData = list(data.iloc[i].name)
        # 添加销售数量
        sale_num = int(data.iloc[i]['主营业务收入'] / unitPrice[i])
        eachData.append(sale_num)
        # 因为将销售数量取整后产生误差，所以用取整后的值重新算一遍
        # 修改在need_split的值会影响到data的值
        need_split[i] = sale_num * unitPrice[i]
        #  添加主营业务收入
        eachData.append(data.iloc[i]['主营业务收入'])
        # 添加销项税额,eachData[3]是税率
        afterTax = data.iloc[i]['主营业务收入'] * eachData[3]
        eachData.append(afterTax)
        # 含税销售额(净额)
        eachData.append(data.iloc[i]['主营业务收入'] + afterTax)
        eachData = pd.Series(eachData, index=need_col)
        # 每一个产品经过了num组分割后的值小于阈值，则添加num组
        split_result = integrate(eachData, count[i], split_result, i)
    # 返回经过所有行的拆分后的拆分结果
    return split_result


"""
    整合数据并导出
"""


def integrate(eachData, count, split_result, i):
    for k in range(count):
        # 这里我觉得健壮性不够强
        if k == count - 1:
            # 要销售数量全部加起来等于原数据的值,
            eachData[4] = data['销售数量'][i] - eachData[4] * float((count - 1))
            # 主营业务收入要等于销售数量*单价
            eachData[5] = eachData[4] * unitPrice[i]
            # 下面两项有0.01的误差
            # 同时也要改变销项税额
            eachData[6] = eachData[5] * eachData[3]
            # 同时也有含税销售额(净额)
            eachData[7] = eachData[5] + eachData[6]
        # DataFrame添加后不改变原来的dataframe，返回的是添加后的dataframe
        split_result = split_result.append(eachData, ignore_index=True)
        #  新一轮的开始
    num = 0
    # 返回添加一轮后的拆分结果
    return split_result


"""
    计算出差异值
"""


def diff(data, split_result):
    global count
    # 拆分前销项税额的总值
    total_afterTax = data['销项税额']
    # 拆分后销项税额的总值
    total_split_result = split_result.groupby(['客户名称', '产品号', '客户', '税率']).sum()
    split_afterTax = total_split_result['销项税额']
    diff_result = list()
    # 计算差异值
    for i in range(len(total_afterTax)):
        difference = split_afterTax[i] - total_afterTax[i]
        diff_result.append([difference] * count[i])
    return list(flatten(diff_result))


"""
    对list降维
"""


def flatten(data):
    for each in data:
        if not isinstance(each, list):
            yield each
        else:
            yield from flatten(each)


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    input_file_name = './split/2.xls'
    output_file_name = './split/final_result.xls'
    # 需要这些列，其余的不要
    need_col = pd.Index(['客户名称', '产品号', '客户', '税率', '销售数量', '主营业务收入', '销项税额', '含税销售额(净额)'])
    # 设置阈值
    value = 30000
    # 存放单价的值
    unitPrice = list()
    # 存储每一组需要拆分成多少小组的值
    count = list()
    # 经过分组整合后的数据
    data = aggergate(input_file_name)
    # 取得是‘主营业务收入’,对这一列数据进行判断
    need_split = data['主营业务收入']
    # 创建个空的dataframe，以存放拆分后的结果
    split_result = pd.DataFrame(columns=need_col)
    # 拆分数据,返回的是拆分后的结果
    split_result = split(data, split_result, need_split)
    # 计算出拆分前与拆分后的差异值
    diff_result = diff(data, split_result)
    # 将差异值添加到数据表中
    split_result['差异'] = diff_result
    col = split_result.columns
    #  以下两行是用来构建层次多级索引用的，使输出的excel表格更具直观性(这里我觉得处理得也不是很好)
    final_result = pd.DataFrame(split_result.values[:, 3:],
                                index=[split_result.values[:, 0], split_result.values[:, 1], split_result.values[:, 2]],
                                columns=col[3:])
    # 设置层次多级索引的名称
    final_result.index.names = ['客户名称', '产品号', '客户']
    final_result.to_excel(output_file_name)
    # result.to_excel('result.xls', index=False)
