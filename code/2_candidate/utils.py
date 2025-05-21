import logging
import os
import pickle
import signal
import multitasking
import numpy as np
import pandas as pd
from random import sample
from tqdm import tqdm

max_threads = multitasking.config['CPU_CORES'] # 获取CPU核心数
multitasking.set_max_threads(max_threads) # 设置允许同时运行的最大线程或进程数max_threads
multitasking.set_engine('process') # 设置多任务引擎为进程,'process' 表示使用多进程作为多任务处理的方式。
signal.signal(signal.SIGINT, multitasking.killall) # 设置信号处理函数，处理Ctrl+C信号，杀死所有子进程。

class Logger(object):
    # 定义一个字典 level_relations，
    # 用于映射日志级别字符串到对应的logging模块中的日志级别常量
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(
        self, 
        filename, 
        level='debug',
        fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    ):
        """
        :param filename: 日志文件名
        :param level: 日志级别，可选值为debug, info, warning, error, crit
        :param fmt:日志格式
        """
        self.logger = logging.getLogger(filename) # 创建一个日志记录器，名称与 filename 相同
        format_str = logging.Formatter(fmt) # 创建一个格式化器，用于指定日志的输出格式。
        self.logger.setLevel(self.level_relations.get(level)) # 根据传入的 level 获取对应的日志级别值。

        sh = logging.StreamHandler() # 创建一个日志处理器，用于将日志输出到控制台
        sh.setFormatter(format_str) # 设置日志格式，使输出到控制台的日志符合 fmt 规定的格式。

        # 创建一个文件处理器：
        # filename=filename：指定日志文件路径。
        # encoding='utf-8'：确保日志文件使用 UTF-8 编码。
        # mode='a'：以 追加模式 打开文件，防止覆盖已有日志。
        th = logging.FileHandler(filename=filename, encoding='utf-8', mode='a')
        th.setFormatter(format_str) # 设置日志格式，使输出到文件的日志符合 fmt 规定的格式。

        self.logger.addHandler(sh) # 将 sh 绑定到 logger，使日志信息可以输出到 控制台。
        self.logger.addHandler(th) # 将 th 绑定到 logger，使日志信息可以存储到 日志文件。
