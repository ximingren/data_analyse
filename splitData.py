# -*- coding: UTF-8 -*-
from multiprocessing.pool import Pool

import pandas as pd
from multiprocessing import Process, cpu_count, Manager
import os

def split(row_data, result_col,need_result_col):
    """
    拆分数据，一行一行处理
    :param row_data: 预处理过的数据总中每一行的数据
    :param result_col: 拆分结果集的列索引
    :param need_split_col:拆分依据的数据值
    :return: each_split_result:每一个进程的拆分结果集
    """
    global split_value,each_split_result  # 使其变为全局变量，能够在另外一个函数中用到
    eachData = list(row_data.name)  # 因为前面分组过了，所以这里输出来的是分组层次索引的数值
    # 加上相应列的值,先取出第i行,然后取出相应的列
    eachData.append(row_data['客户'])
    eachData.append(row_data['利润中心'])
    eachData.append(row_data['纳税人识别号'])
    eachData.append(row_data['产品号'])
    eachData.append(row_data['税率'])
    eachData.append(row_data['计'])
    param_len = len(eachData)  # 固定参数长度
        # 如果数值为0，则后续的不用计算具体的值
    if row_data['销售数量'] == 0:
        # 单价为0
        unitPrice=0
        # 只用加一行
        count=1
        # 加上相应的列
        eachData.append(row_data['销售数量'])
        eachData.append(row_data['主营业务收入'])
        eachData.append(row_data['销项税额'])
        eachData.append(row_data['含税销售额(净额)'])
        # 如果数值不为0，则计算相应的值
    else:
        unitPrice=row_data['主营业务收入'] / row_data['销售数量']  # 计算每件物品的单价,主营业务收入/销售数量=单价
        split_value = need_result_col  # split_value是最后的拆分值
        # 判断是否大于阈值
        if split_value > split_threshold:
            #  若一直大于则一直二分下去
            while (split_value > split_threshold):
                split_value = split_value / 2.0
        count=int(row_data['主营业务收入'] / split_value)  # 计算要分成多少组才能小于阈值,总值/划分结果值=组数
        sale_num = int(split_value / unitPrice)  # 计算出销售数量,销售数量=拆分值/单价
        afterTax = float('%.2f' % (split_value * row_data['税率']))  # 计算出销项税额,主营业务收入*税率=销项税额
        eachData.append(sale_num)  # 添加销售数量
        eachData.append(float('%.2f' % (split_value)))  # 添加主营业务收入，保留2位小数
        eachData.append(afterTax)  # 添加销项税额
        eachData.append(float('%.2f' % (split_value + afterTax)))  # 主营业务收入+销项税额=含税销售额(净额)，并天津爱
    eachData = pd.Series(eachData, index=result_col)  # 拆分值的每一行
    each_split_result=integrate(row_data, eachData, split_value, each_split_result, param_len, unitPrice, count) #整合数据
    # # 返回经过所有行的拆分后的拆分结果集

def integrate(row_data, eachData, split_value, each_split_result, param_len, unitPrice, count):
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
    for k in range(count):
        """计算保留2位小数后和原始数据的误差,累加保留2位小数后和划分结果值的差值
           eachData[param_len]是销售数量,所以each[param_len+1]是主营业务收入
           累加每一组的保留小数后的主营业务收入和保留前的误差
        """
        income_mistakes = income_mistakes + eachData[param_len + 1] - split_value
        # 到了最后一列要根据原始数据和划分后的数据比较，更改数据
        if k == count - 1:
            # 要销售数量全部加起来等于原数据的值,总销售数量-总的划分销售数量=最后一行的销售数量
            eachData[param_len] = int(row_data['销售数量'] - eachData[param_len] * (count - 1))
            # 销售数量乘于单价，再加上差值,即为最后一行的主营业务收入
            eachData[param_len + 1] = float('%.2f' % (eachData[param_len] * unitPrice + income_mistakes))
            # 同时也要改变销项税额，主营业务收入乘于税率
            eachData[param_len + 2] = float('%.2f' % (eachData[param_len + 1] * row_data['税率']))
            # 同时也有含税销售额(净额)
            eachData[param_len + 3] = float('%.2f' % (eachData[param_len + 2] + eachData[param_len + 1]))
        # DataFrame添加后不改变原来的DataFrame，返回的是添加后的dataframe
        each_split_result=each_split_result.append(eachData,ignore_index=True)
        # split_result = split_result.append(eachData, ignore_index=True)
    return each_split_result
    # 返回添加一轮后的拆分结果


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


def fillna_reduce_table(data, summary_data, merged_raw_data):
    """
    核减之后的表中有nan值，将其补充完整。
    补充完后缺失部分列的值，将其补充完整。
    :param data:  核减后的数据
    :param summary_data: 核减后填充nan值所用的参考数据
    :param merged_raw_data: 用以补充客户、产品等列的参考数据
    :return: 经过补充填充后的数据
    """
    data = data.combine_first(summary_data)  # 对未对应得上的数据进行填充处理
    merged_raw_data = merged_raw_data.reindex(columns=result_col).set_index(['购货单位', '产品'])  # 对未分组的数据进行处理，设置列排序，设置层次索引
    data = data.reindex(columns=result_col.drop(['购货单位', '产品']))  # 设置列排序（其实是新添多个列）
    group_index2 = ['购货单位', '产品','客户', '利润中心','纳税人识别号','产品号', '税率','计']
    merged_raw_data = merged_raw_data.groupby(group_index2).sum().reset_index().set_index(group_index)
    fill_data = data.combine_first(merged_raw_data)  # 两个表填充
    return fill_data


def pre_process(raw_data, reduce_data, reference_data, need_col, need_reference_col, group_index):
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
    data = summary_data.sub(reduce_data)  # 核减操作
    fill_data = fillna_reduce_table(data, summary_data, merged_raw_data)  # 进行补充nan值和添加缺失的列
    fill_data.to_excel('./result/before_split_total_data.xls')  # 导出拆分之前的总数据，但是核减过后的
    return fill_data


def process_invoice(shop_unit, tax,income,properties):
    """
    添加发票张数的时候要根据限制值返回具体的张数标志
    :param shop_unit: 每一行购货单位
    :param tax: 每一行税率
    :return: 发票张数标志
    """
    global last_shop_unit, last_tax, invoice_group_num,each_group_incomes

    #  购货单位不同，就加一张发票
    if last_shop_unit != shop_unit:
        invoice_group_num += 1 #加一张发票
        last_shop_unit = shop_unit
        last_tax = tax
        each_group_incomes = 0 # 新的一张发票的收入和归零
    # 购货单位相同、但是税率不同，加一张发票
    elif last_tax != tax:
        invoice_group_num += 1
        last_tax = tax
        each_group_incomes = 0
    # 如果主营业务收入大于合并阈值，加一张发票
    if properties == '普票':
        if each_group_incomes+income > common_merge_threshold:
            invoice_group_num += 1
            each_group_incomes = 0
    if properties == '专票':
        if each_group_incomes+income > special_merge_threshold:
            invoice_group_num += 1
            each_group_incomes = 0
    each_group_incomes+=income #累计收入和
    return "A" + str(invoice_group_num)


def add_invoice_num(diff_result):
    """
    添加发票张数这一列
    :param diff_result: 添加完差异列后的总数据表
    :return: 添加完发票张数后的总数据表
    """
    global last_shop_unit, last_tax, invoice_group_num,each_group_incomes
    last_shop_unit = last_tax = None  # 上一行的购货单位，上一行的税率
    invoice_group_num = 0  # 发票张数的数字标志
    each_group_incomes=0 # 每一张发票主营业务收入的和
    #  apply函数将每一行的购货单位和税率当作参数传入函数中进行处理
    diff_result['发票张数'] = diff_result.apply(lambda x: process_invoice(x['购货单位'], x['税率'],x['主营业务收入'],x['发票性质']),axis=1)
    return diff_result


def add_invoice_properties(tax, cumstomer):
    """
    增加发票性质这一列
    :param tax: 每一行的税率
    :param cumstomer: 每一行的客户代码
    :return: 发票性质标志
    """
    if tax == 0.16:
        return "专票"
    if tax == 0.10:
        if special_tickets_refernce_customers.isin([cumstomer]).any():
            return "专票"
        return "普票"
def use_multiprocess_split():
    global  each_split_result,split_result_list
    each_split_result = pd.DataFrame(columns=result_col)  # 创建个空的DataFrame，以存放拆分后的结果
    split_result_list = Manager().list()
    pool = Pool(4)
    for x in range(cpu_count()):
        split_position = int(len(data) / cpu_count())  # 划分数据集，分给多进程任务进行处理
        each_process_data = data.iloc[split_position * x:split_position * (x + 1)]  # 每一个进程用到的数据
        if x == 3:
            each_process_data = data.iloc[split_position * x:]
        pool.apply_async(each_process_split, args=(each_process_data,))  # 非阻塞执行
    pool.close()
    pool.join()
    split_result = pd.DataFrame(columns=result_col)
    for result in split_result_list:
        split_result = split_result.append(result)
    return split_result
def each_process_split(each_process_data):
    """
    多进程用到的target函数
    :param data: 每一个进程所用的数据
    :return:
    """
    each_process_data.apply(lambda x: split(x, result_col, x['主营业务收入']), axis=1)
    split_result_list.append(each_split_result)  # 共享变量中加入每一个进程的拆分结果集
if __name__ == '__main__':
    """下面的配置变量有点多,找机会改一下看怎么弄，怎么简化它"""
    pd.set_option('display.width', 200)
    reduce_file = './rawdata/核减表（测试）1 (1).XLSX'  # 核减表
    rawData_file = "./rawdata/汇总表（测试）1 (1).XLSX"  # 原始总数据表
    reference_file = './rawdata/参照表(1).xlsx'  # 参照表,用以添加购货单位和利润中心
    special_tickets_reference_file = "./rawdata/专票客户.xlsx"  # 用以修改税率为.1中的普票的参照表
    create_directory("./result")  # 创建存放结果集的目录
    split_result_file = './result/final_result10.xls'
    special_merge_threshold=split_threshold = 30000  # 设置专票合并阈值和划分阈值
    common_merge_threshold = 300000  # 设置普票合并阈值
    reduce_data = pd.read_excel(reduce_file)  # 核减的数值
    raw_data = pd.read_excel(rawData_file)  # 为处理过的总数据
    reference_data = pd.read_excel(reference_file).fillna(0)  # 参照表的数据,并将nan值填充0
    special_tickets_refernce_data = pd.read_excel(special_tickets_reference_file) # 专票参照数据
    need_col = pd.Index(['客户', '产品号', '产品', '税率','计','销售数量', '主营业务收入', '销项税额', '含税销售额(净额)'])  # 处理raw数据需要用到的列
    need_reference_col = pd.Index(['客户', '购货单位', '利润中心','纳税人识别号'])  # 参照表需要用到的列,将购货单位、利润中心、纳税人识别号添加进原始数据表中
    result_col = pd.Index(['购货单位', '产品', '客户', '利润中心','纳税人识别号','产品号', '税率','计','销售数量', '主营业务收入', '销项税额', '含税销售额(净额)'])  # 结果集的列
    group_index = ['购货单位', '产品']  # 分组依据的索引
    print("进行数据预处理")
    data = pre_process(raw_data, reduce_data, reference_data, need_col, need_reference_col, group_index)  # 预处理操作
    # 取得是‘主营业务收入’,对这一列数据进行判断
    need_split_col = data['主营业务收入']
    # 拆分数据,返回的是拆分后的结果
    print("进行数据拆分")
    split_result=use_multiprocess_split()
    # 计算出拆分前与拆分后的差异值,返回的是有差异列的结果集
    print("进行计算差异")
    diff_result = diff(split_result)
    print("进行添加发票性质一列")
    special_tickets_refernce_customers = special_tickets_refernce_data['客户'] #专票参照数据表中的客户代码
    diff_result['发票性质'] = diff_result.apply(lambda x: add_invoice_properties(x['税率'], x['客户']), axis=1)
    print("进行添加发票张数一列")
    diff_result = add_invoice_num(diff_result)
    diff_result = diff_result.set_index(group_index)  # 设置为多级索引
    diff_result.to_excel(split_result_file)
