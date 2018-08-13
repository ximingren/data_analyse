import os
from multiprocessing.pool import Pool

import pandas as pd
import xlrd
from multiprocessing import Lock, Manager


class Table():
    def __init__(self, raw_data_file, reduce_data_file, reference_data_file, special_customers_file, sheet_name):
        self.sheet_name = sheet_name
        self.raw_data = pd.read_excel(raw_data_file, self.sheet_name)
        self.reduce_data = pd.read_excel(reduce_data_file, self.sheet_name)
        self.reference_data = pd.read_excel(reference_data_file).fillna(0)
        self.special_customers = pd.read_excel(special_customers_file)['客户']
        self.result_col = pd.Index(
            ['购货单位', '产品', '客户', '利润中心', '纳税人识别号', '产品号', '税率', '计', '销售数量', '主营业务收入', '销项税额', '含税销售额(净额)'])  # 结果集的列
        self.need_col = pd.Index(['客户', '产品号', '产品', '税率', '计', '销售数量', '主营业务收入', '销项税额', '含税销售额(净额)'])  # 处理raw数据需要用到的列
        self.need_reference_col = pd.Index(['客户', '购货单位', '利润中心', '纳税人识别号'])  # 参照表需要用到的列,将购货单位、利润中心、纳税人识别号添加进原始数据表中
        self.group_index = ['购货单位', '产品']

    def prepare_env(self, need_split_col, split_threshold, common_merge_threshold):
        pd.set_option('display.width', 200)
        self.create_directory("./result")
        self.need_split_col = need_split_col
        self.split_threshold = split_threshold  # 设置专票合并阈值和划分阈值
        self.merge_threshold = {'专票': split_threshold, '普票': common_merge_threshold}
        self.each_split_result = pd.DataFrame(columns=self.result_col)  # 创建个空的DataFrame，以存放拆分后的结果

    def create_directory(self, path):
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

    def pre_process(self):
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
        self.raw_data = self.raw_data[self.need_col].fillna(0)  # 取出想要的列,并将nan值填充0
        self.reduce_data = self.reduce_data[self.need_col].fillna(0)  # 取出想要的列,并将nan值填充0
        # 从参照表中取出想要的列，然后和原始数据和核减数据合并
        #  合并两个表,即是将raw数据表和reduce数据表添加购货单位,利润中心,纳税人识别号。依据客户这一列进行合并
        self.merged_raw_data = self.raw_data.merge(self.reference_data[self.need_reference_col], left_on='客户',
                                                   right_on='客户', how='left')
        self.merged_reduce_data = self.reduce_data.merge(self.reference_data[self.need_reference_col], left_on='客户',
                                                         right_on='客户',
                                                         how='left').dropna()
        # 先进行分组了汇总了再进行核减
        self.summary_data = self.merged_raw_data.groupby(self.group_index).agg(
            {'销售数量': 'sum', '主营业务收入': 'sum', '销项税额': 'sum', '含税销售额(净额)': 'sum'})  # 分组操作并聚合运算
        self.reduce_data = self.merged_reduce_data.groupby(self.group_index).agg(
            {'销售数量': 'sum', '主营业务收入': 'sum', '销项税额': 'sum', '含税销售额(净额)': 'sum'})  # 分组操作并聚合运算
        self.reduced_nan_data = self.summary_data.sub(self.reduce_data)  # 核减操作,reduced_data是减完的数据，但是含有nan值
        self.fillna_reduced_table()  # 进行填充nan值
        self.add_lack_columns()  # 添加缺失的列

    def fillna_reduced_table(self):
        """
        核减之后的表中有nan值，将其补充完整。
        补充完后缺失部分列的值，将其补充完整。
        :param data:  核减后的数据
        :param summary_data: 核减后填充nan值所用的参考数据
        :param merged_raw_data: 用以补充客户、产品等列的参考数据
        :return: 经过补充填充后的数据
        """
        self.reduced_data = self.reduced_nan_data.combine_first(self.summary_data)  # 对未对应得上的数据进行填充处理，用的是合并了参照表和分组后的数据

    def add_lack_columns(self):
        """
        添加缺失的列
        :return:
        """
        self.merged_raw_data = self.merged_raw_data.reindex(columns=self.result_col)  # 对未分组的数据进行处理，设置列排序，设置层次索引
        self.merged_raw__duplicate_data = self.merged_raw_data.drop_duplicates(self.group_index)
        self.merged_raw__duplicate_data.set_index(self.group_index, inplace=True)
        self.reduced_data = self.reduced_data.reindex(columns=self.result_col.drop(['购货单位', '产品']))  # 设置列排序（其实是新添多个列）
        self.ripe_data = self.reduced_data.combine_first(self.merged_raw__duplicate_data)
        self.ripe_data = self.ripe_data.dropna()

    def each_process_split(self, each_process_data):
        """
        多进程用到的target函数
        :param each_process_data: 每一个进程所用的数据
        :return:
        """
        each_process_data.apply(lambda x: self.split(x), axis=1)

    def split(self, row_data):
        """
        拆分数据，一行一行处理
        :param row_data: 预处理过的数据总中每一行的数据
        :return:
        """
        # global split_value, each_split_result  # 使其变为全局变量，能够在另外一个函数中用到
        self.eachData = list(row_data.name)  # 因为前面分组过了，所以这里输出来的是分组层次索引的数值
        # 加上相应列的值,先取出第i行,然后取出固定不变的列
        self.eachData.append(row_data['客户'])
        self.eachData.append(row_data['利润中心'])
        self.eachData.append(row_data['纳税人识别号'])
        self.eachData.append(row_data['产品号'])
        self.eachData.append(row_data['税率'])
        self.eachData.append(row_data['计'])
        self.param_len = len(self.eachData)  # 固定参数长度
        # 如果数值为0，则后续的不用计算具体的值
        if row_data['销售数量'] == 0:
            # 单价为0
            unitPrice = 0
            # 只用加一行
            count = 1
            # 加上相应的列
            self.eachData.append(row_data['销售数量'])
            self.eachData.append(row_data['主营业务收入'])
            self.eachData.append(row_data['销项税额'])
            self.eachData.append(row_data['含税销售额(净额)'])
            # 如果数值不为0，则计算相应的值
        else:
            self.unitPrice = row_data['主营业务收入'] / row_data['销售数量']  # 计算每件物品的单价,主营业务收入/销售数量=单价
            self.split_value = row_data[self.need_split_col]  # split_value是最后的拆分值
            # 判断是否大于阈值
            if self.split_value > self.split_threshold:
                #  若一直大于则一直二分下去
                self.split_value=self.split_threshold
                # while (self.split_value > self.split_threshold):
                #     self.split_value = self.split_value / 2.0
            self.split_counts = int(row_data['主营业务收入'] / self.split_threshold)+1  # 计算要分成多少组才能小于阈值,总值/划分结果值=组数
            self.sale_num = int(self.split_value / self.unitPrice)  # 计算出销售数量,销售数量=拆分值/单价
            self.afterTax = self.split_value * row_data['税率']  # 计算出销项税额,主营业务收入*税率=销项税额
            self.eachData.append(self.sale_num)  # 添加销售数量
            self.eachData.append(float('%.2f' % (self.split_value)))  # 添加主营业务收入，保留2位小数
            self.eachData.append(float('%.2f' % (self.afterTax)))  # 添加销项税额
            self.eachData.append(float('%.2f' % (self.split_value + self.afterTax)))  # 主营业务收入+销项税额=含税销售额(净额)，并天津爱
        self.eachData = pd.Series(self.eachData, index=self.result_col)  # 拆分值的每一行
        self.integrate(row_data)  # 整合数据
        # # 返回经过所有行的拆分后的拆分结果集

    def integrate(self, row_data):
        """
        将拆分行一行一行得整合出来
        :param row_data:  行数据
        :return:
        """
        # self.income_mistakes = 0  # 主营业务收入误差,保留2位小数后产生的误差
        for k in range(self.split_counts):
            """计算保留2位小数后和原始数据的误差,累加保留2位小数后和划分结果值的差值
               eachData[param_len]是销售数量,所以each[param_len+1]是主营业务收入
               累加每一组的保留小数后的主营业务收入和保留前的误差
            """
            # 到了最后一列要根据原始数据和划分后的数据比较，更改数据
            if k == self.split_counts - 1:
                # 要销售数量全部加起来等于原数据的值,总销售数量-总的划分销售数量=最后一行的销售数量
                self.eachData[self.param_len] = row_data['销售数量'] - self.eachData[self.param_len] * (self.split_counts - 1)
                # 销售数量乘于单价，再加上差值,即为最后一行的主营业务收入
                self.eachData[self.param_len + 1] = float(
                    '%.2f' % (row_data['主营业务收入']- self.eachData[self.param_len+1] * (self.split_counts - 1)))
                # 同时也要改变销项税额，主营业务收入乘于税率
                self.eachData[self.param_len + 2] = float('%.2f' % (self.eachData[self.param_len + 1] * row_data['税率']))
                # 同时也有含税销售额(净额)
                self.eachData[self.param_len + 3] = float(
                    '%.2f' % (self.eachData[self.param_len + 2] + self.eachData[self.param_len + 1]))
            self.each_split_result = self.each_split_result.append(self.eachData, ignore_index=True)
    def diff(self):
        """
        计算出差异值
        """
        self.each_split_result['销项税额(计算)'] = self.each_split_result.apply(lambda x,: x['主营业务收入'] * x['税率'],
                                                                          axis=1)  # 计算出来的销项税额
        self.each_split_result['差异'] = self.each_split_result.apply(lambda x: x['销项税额(计算)'] - x['销项税额'],
                                                                    axis=1)  # 计算出来的销项税额和未计算的销项税额的差值

    def process_invoice(self, shop_unit, tax, income, properties):
        """
        添加发票张数的时候要根据限制值返回具体的张数标志
        :param shop_unit: 每一行购货单位
        :param tax: 每一行税率
        :return: 发票张数标志
        """
        #  购货单位不同，就加一张发票
        if self.last_shop_unit != shop_unit or self.each_invoice_incomes + income > self.merge_threshold[properties]:
            self.invoice_counts += 1  # 加一张发票
            self.each_invoice_incomes = 0  # 新的一张发票的收入和归零
            self.last_shop_unit = shop_unit
            self.last_tax = tax
            # 购货单位相同、但是税率不同，加一张发票
        elif self.last_tax != tax or self.each_invoice_incomes + income > self.merge_threshold[properties]:
            self.invoice_counts += 1  # 加一张发票
            self.each_invoice_incomes = 0  # 新的一张发票的收入和归零
            self.last_tax = tax
        # 如果主营业务收入大于合并阈值，加一张发票
        self.each_invoice_incomes += income  # 累计收入和
        return "A" + str(self.invoice_counts)

    def add_invoice_num(self):
        """
        添加发票张数这一列
        分两轮，购货单位和税率同样的为一张发票，若同张发票的收入和大于合并阈值则增加一张发票
        :param diff_result: 添加完差异列后的总数据表
        :return: 添加完发票张数后的总数据表
        """
        self.last_shop_unit = self.last_tax = None  # 上一行的购货单位，上一行的税率
        self.invoice_counts = 0  # 发票的张数
        self.each_invoice_incomes = 0  # 每一张发票主营业务收入的和
        #  apply函数将每一行的购货单位和税率当作参数传入函数中进行处理
        self.each_split_result['发票张数'] = self.each_split_result.apply(
            lambda x: self.process_invoice(x['购货单位'], x['税率'], x['主营业务收入'], x['发票性质']), axis=1)

    def add_invoice_properties(self, tax, cumstomer):
        """
        增加发票性质这一列
        :param tax: 每一行的税率
        :param cumstomer: 每一行的客户代码
        :return: 发票性质标志
        """
        if tax == 0.16:
            return "专票"
        if tax == 0.10:
            if self.special_customers.isin([cumstomer]).any():
                return "专票"
            return "普票"

    def add_invoice(self):
        self.each_split_result['发票性质'] = self.each_split_result.apply(
            lambda x: self.add_invoice_properties(x['税率'], x['客户']), axis=1)  # 执行增加发票性质的函数
        self.add_invoice_num()  # 执行增加发票数量的函数


def process_data_main(sheet_name, lock):
    """
    数据处理的主程序
    :param sheet_name: sheet的名称
    :return:
    """
    table = Table(rawData_file, reduce_file, reference_file, special_tickets_reference_file,
                  sheet_name)  # 实例化sheet表，这个对象是数据处理的载体
    table.prepare_env("主营业务收入", 30000, 300000)  # 准备工作做好,第一个参数是划分数据依据的列，第二个参数是划分阈值，第三个参数是普票合并阈值
    print("对%s表进行数据预处理" % table.sheet_name)
    table.pre_process()  # 对数据进行预处理
    print("对%s表进行数据划分" % table.sheet_name)
    table.each_process_split(table.ripe_data)  # 划分数据
    print("对%s表进行计算差异值" % table.sheet_name)
    table.diff()  # 计算差异
    print("对%s表进行添加发票相关的列" % table.sheet_name)
    table.add_invoice()  # 增添发票性质和发票数目这两列
    lock.acquire()  # 获取锁
    split_data_list.append(table.ripe_data)
    final_data_list.append(table.each_split_result)
    sheet_names_list.append(table.sheet_name)
    lock.release()  # 释放锁


def export_data(no_split_file, splited_file):
    """
    导出数据表
    :param no_split_file:  未拆分的数据
    :param splited_file:  拆分结果集
    :return:
    """
    try:
        no_split_writer = pd.ExcelWriter(no_split_file)
        split_writer = pd.ExcelWriter(splited_file)
        for x in range(len(split_data_list)):
            sheet_name = sheet_names_list[x]
            print("导出%s表" % sheet_name)
            split_data_list[x].to_excel(no_split_writer, sheet_name=sheet_name)
            final_data_list[x] = final_data_list[x].set_index(['购货单位', '产品'])
            final_data_list[x].to_excel(split_writer, sheet_name=sheet_name)
        no_split_writer.save()
        split_writer.save()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    try:
        reduce_file = r'.\rawdata\核减表（测试）(1).XLSX'  # 核减表
        rawData_file = r".\rawdata\汇总表（测试）(1).XLSX"  # 原始总数据表
        reference_file = r'.\rawdata\参照表 (2)4.xlsx'  # 参照表,用以添加购货单位和利润中心
        special_tickets_reference_file = r".\rawdata\专票客户4.xlsx"  # 用以修改税率为0.1中的普票的参照表
        manager = Manager()
        split_data_list = manager.list()  # 存放未拆分前的总数据
        final_data_list = manager.list()  # 存放最后的结果集
        sheet_names_list = manager.list()
        book = xlrd.open_workbook(rawData_file)  # 获取excel文件工作簿
        lock = Lock()
        pool = Pool(len(book.sheet_names()))  # 构建进程池
        for sheet_name in book.sheet_names():
            pool.apply_async(process_data_main(sheet_name, lock))  # 每个sheet表格用单独的进程来处理
        pool.close()  # 关闭进程池
        pool.join()  # 阻止当前进程,直到调用join方法的那个进程执行完，再执行当前进程.
        export_data(r".\result\before_split_total_data.xls", r".\result\final_result.xls")  # 导出数据表
    except Exception as e:
        print(e)
