# -*- coding:utf-8 -*-
from copy import deepcopy
from multiprocessing import Pool, Manager
import os
import pandas as pd

"""
    本脚本的原始数据是csv文件、且第一个sheet是等待处理的原始数据表
    采用多进程和读取存储csv格式文件的方法防止内存崩溃，若直接读取xls文件，会报MemoryError错误
    建议所有的数据文件在处理前转换为csv格式，导出结果集的文件后可以手动将csv转换为xls等格式
    建议小excel文件用脚本读取即可，大excel文件存入数据库中进行处理会更加方便
"""

def count_posts(name, value, reply_post):
    """
    统计回帖人的累计
    :param name: 发帖人名称
    :param value: 回帖人列表
    :param reply_post:  回帖人dict
    :return:
    """
    global index, each_final_result
    try:
        print("进程%d正在对%s进行统计分析" % (os.getpid(), name))
        each_reply_post=deepcopy(reply_post) # 深复制
        for x in value:
            # 如果回帖人在回帖人dict的key中则执行下面代码
            if x in each_reply_post.keys():
                # 下面的这个x==x是为了除去nan值
                if x == x:
                    each_reply_post[x] += 1 # 在一个帖子下，回帖人重复出现了则加一
        each_final_result[name] = list(each_reply_post.values()) # 在最后的结果集中加入数值
    except Exception as e:
        print(e)

def process_main(each_data, each_index, reply_post):
    """
    统计处理数据的主程序
    :param each_data: 每一个进程要处理的数据
    :param index:  回帖人的汇总
    :param reply_post: 回帖人dict
    :return:
    """
    global each_final_result
    try:
        each_final_result = pd.DataFrame(index=pd.Index(each_index)) # 构建每一个进程存放最后结果集的变量
        each_data.apply(lambda x: count_posts(x.name, x.values, reply_post)) # 对each_data的每一列进行迭代，传入列的名称，列的数值和回帖人的dict
        print("导出表")
        each_final_result.to_csv(str(os.getpid())+".csv") # 导出结果集为csv文件
    except Exception as e:
        print(e)


if __name__ == '__main__':
    pd.set_option('display.width', 500) # 设置显示宽度
    print("读取数据文件")
    post_data = pd.read_csv('1.csv', encoding='ANSI') #取得原数据
    index=post_data.columns # 结果集的行索引和列索引是一致的
    pool = Pool(4) # 构建进程池
    each_process_length = int(len(post_data.columns) / 4) # 每一个进程需要处理数据的大小
    for x in range(os.cpu_count()):
        reply_post={} # 回帖人dict
        if x != os.cpu_count() - 1:
            each_index=index[each_process_length * x:each_process_length * (x + 1)] # 前三个进程的行索引
            each_data = post_data.iloc[:, each_process_length * x:each_process_length * (x + 1)] # 前三个进程所要处理的数据
        else:
            each_index=index[each_process_length * x:] # 最后一个进程的行索引
            each_data = post_data.iloc[:, each_process_length * x:] # 最后一个进程所要处理的数据
        for x in each_index:
            reply_post.setdefault(x, 0) # 构建回帖人dict
        pool.apply_async(process_main, args=(each_data, each_index, reply_post)) # 非阻塞执行处理数据的主函数，传入全局变量使之成为进程的独立变量
    pool.close() # 关闭进程池
    pool.join() # 等待子进程的完成
    print("数据处理完成")
