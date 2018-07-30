# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np



"""
    拆分数据
"""


def split(data, split_result, need_split):
    global count
    global split_value
    # 对数据的每一行进行判断，若数据大于阈值则需要被分割
    for i in range(len(need_split)):
        # 第i行的名称，因为前面整合的时候分组了，所以这里输出来的是分组的数值
        eachData = list(data.iloc[i].name)
        # 加上相应的值
        eachData.append(data.iloc[i]['客户'])
        eachData.append(data.iloc[i]['利润中心'])
        eachData.append(data.iloc[i]['产品号'])
        eachData.append(data.iloc[i]['税率'])
        param_len = len(eachData)
        # 如果数值为0，则后续的不用计算具体的值，直接赋值为0
        if data['主营业务收入'][i] == 0:
            for k in range(len(group_index) - param_len):
                eachData.append(0)
            count.append(1)
            unitPrice.append(0)
        # 如果数值不为0，则计算相应的值
        else:
            # 计算每件物品的单价,主营业务收入/销售数量=单价
            unitPrice.append(data['主营业务收入'][i] / data['销售数量'][i])
            # 判断是否大于阈值,split_value是最后的拆分值
            split_value = need_split[i]
            if split_value > value:
                #  若一直大于则一直二分下去
                while (split_value > value):
                    # 计算二分了多少次
                    split_value = split_value / 2.0
            # 计算要分成多少组才能小于阈值,总值/划分结果值=组数
            count.append(int(data['主营业务收入'][i] / split_value))
            # 添加销售数量
            sale_num = int(split_value / unitPrice[i])
            eachData.append(sale_num)
            #  添加主营业务收入
            eachData.append(float('%.2f' % (split_value)))
            # 添加销项税额,主营业务收入*税率=销项税额,税率在group_index的最后一个
            afterTax = float('%.2f' % (split_value * data['税率'][i]))
            eachData.append(afterTax)
            # 主营业务收入+销项税额=含税销售额(净额)
            eachData.append(float('%.2f' % (split_value + afterTax)))
        eachData = pd.Series(eachData, index=final_col)
        # 每一个产品经过了num组分割后的值小于阈值，则添加num组
        split_result = integrate(data, eachData, split_value, split_result, param_len, count[i], i)
    # 返回经过所有行的拆分后的拆分结果
    return split_result


"""
    整合数据并导出
"""


def integrate(data, eachData, split_value, split_result, param_len, count, i):
    # 主营业务收入误差
    income_mistake = 0
    for k in range(count):
        # 计算保留2位小数后和原始数据的误差,累加保留2位小数后和划分结果值的差值
        income_mistake = income_mistake + eachData[param_len + 1] - split_value
        # 到了最后一列要根据原始数据和划分后的数据比较，更改
        if k == count - 1:
            # 要销售数量全部加起来等于原数据的值,总销售数量-总的划分销售数量=最后一行的销售数量
            eachData[param_len] = int(data['销售数量'][i] - eachData[param_len] * (count - 1))
            # 销售数量乘于单价，再加上差值,即为最后一行的销售数量
            eachData[param_len + 1] = float('%.2f' % (eachData[param_len] * unitPrice[i] + income_mistake))
            # 同时也要改变销项税额，主营业务收入乘于税率
            eachData[param_len + 2] = float('%.2f' % (eachData[param_len + 1] * data['税率'][i]))
            # 同时也有含税销售额(净额)
            eachData[param_len + 3] = float('%.2f' % (eachData[param_len + 2] + eachData[param_len + 1]))
        # DataFrame添加后不改变原来的dataframe，返回的是添加后的dataframe
        split_result = split_result.append(eachData, ignore_index=True)
        #  新一轮的开始
    # 返回添加一轮后的拆分结果
    return split_result


"""
    计算出差异值
"""

def diff(data, split_result):
    global count
    # 拆分前销项税额的总值
    beforeSplit_total_afterTax = data['销项税额']
    # 拆分后销项税额的总值
    total_split_result = split_result.groupby(group_index).agg({'销项税额': 'sum'})
    afterSplit_total_afterTax = total_split_result['销项税额']
    split_data_index = afterSplit_total_afterTax.index
    diff_result = list()
    # 计算差异值
    for i in range(len(beforeSplit_total_afterTax)):
        difference = afterSplit_total_afterTax.loc[split_data_index[i]] - beforeSplit_total_afterTax[
            split_data_index[i]]
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
    reduce_file = '8.xls'
    rawdata_file = "7.xls"
    reference_file = '参照表 (1).xlsx'
    output_file_name = 'final_result5.xls'
    # 需要这些列，其余的不要
    # 设置阈值
    value = 30000
    # 存放单价的值
    unitPrice = list()
    # 存储每一组需要拆分成多少小组的值
    count = list()
    # 经过分组整合后的数据
    reduce_data = pd.read_excel(reduce_file) #核减的数据
    total_data = pd.read_excel(rawdata_file) #总的数据
    reference_data = pd.read_excel(reference_file).fillna(0)#参照表的数据,并将nan值填充0
    need_col = pd.Index(['客户','产品号','产品', '税率', '销售数量', '主营业务收入', '销项税额', '含税销售额(净额)'])
    need_reference_col = pd.Index(['客户', '购货单位','利润中心'])
    final_col=pd.Index(['购货单位','产品','客户','利润中心','产品号','税率','销售数量', '主营业务收入', '销项税额', '含税销售额(净额)'])
    group_index = ['购货单位','产品']
    total_data = total_data[need_col].fillna(0)  # 删除含有NAN值的行
    reduce_data = reduce_data[need_col].fillna(0)  # 删除含有NAN值的行
    #  合并两个表,即是替换购货单位那里
    add_purchase_total_data = total_data.merge(reference_data[need_reference_col], left_on='客户', right_on='客户', how='left')
    add_purchase_reduce_data = reduce_data.merge(reference_data[need_reference_col], left_on='客户', right_on='客户', how='left').dropna()
    # 分组操作
    summary_data=add_purchase_total_data.groupby(['购货单位','产品']).agg({'销售数量':'sum','主营业务收入':'sum','销项税额':'sum','含税销售额(净额)':'sum'})
    reduce_data=add_purchase_reduce_data.groupby(['购货单位','产品']).agg({'销售数量':'sum','主营业务收入':'sum','销项税额':'sum','含税销售额(净额)':'sum'})
    # 核减操作
    data=(summary_data-reduce_data)
    # 对应不上的没有核减的数据
    no_reduce_data=data[pd.isna(data)['主营业务收入']]
    add_purchase_total_data=add_purchase_total_data.set_index(['购货单位','产品']) #层次索引
    # summary_data=summary_data.set_index(['购货单位','产品'])  # 层次索引
    for x in range(len(no_reduce_data)):
        miss_value=summary_data.loc[no_reduce_data.iloc[x].name]
        data.loc[no_reduce_data.iloc[x].name]=miss_value
        no_reduce_data.iloc[x]=miss_value
    no_reduce_data.to_excel('no_reduce_data.xls')
    data.to_excel('total_before_split_data.xls')
    # 加上客户，产品号，税率,利润中心这些字段
    data['客户']=None
    data['产品号']=None
    data['税率']=None
    data['利润中心']=None
    # 添加漏掉的字段值
    for x in range(len(data)):
        customer=add_purchase_total_data.loc[data.iloc[x].name]['客户'].values[0]
        product_number=add_purchase_total_data.loc[data.iloc[x].name]['产品号'].values[0]
        tax=add_purchase_total_data.loc[data.iloc[x].name]['税率'].values[0]
        profit=add_purchase_total_data.loc[data.iloc[x].name]['利润中心'].values[0]
        data.loc[data.iloc[x].name,'客户']=customer
        data.loc[data.iloc[x].name,'利润中心']=profit
        data.loc[data.iloc[x].name,'产品号']=product_number
        data.loc[data.iloc[x].name,'税率']=tax
    data_copy=data.copy() #复制数据,以便计算差异时用到
    # 取得是‘主营业务收入’,对这一列数据进行判断
    need_split = data['主营业务收入']
    # 创建个空的dataframe，以存放拆分后的结果
    split_result = pd.DataFrame(columns=final_col)
    # 拆分数据,返回的是拆分后的结果
    split_result = split(data, split_result, need_split)
    # 计算出拆分前与拆分后的差异值
    diff_result = diff(data_copy, split_result)
    # # 将差异值添加到数据表中
    split_result['差异'] = diff_result
    split_result.to_excel(output_file_name,index=False)

