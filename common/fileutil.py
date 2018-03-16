# -*- coding: utf-8 -*-  

'''
文件读取、存储通用类
'''

import csv
import codecs

class CSVWriter():
    '''CSV文件存储通用类
    '''
    
    '''构造函数，需要提供文件路径
    '''
    def __init__(self, filePath):
        self.filePath = filePath
        self.csvfile = file(filePath, 'wb')
        self.csvfile.write(codecs.BOM_UTF8)
        self.writer = csv.writer(self.csvfile)

    '''写一行数据
    '''
    def writeRow(self, row):
        self.writer.writerow([row])
        self.csvfile.flush()
        
    '''关闭
    '''
    def close(self):
        self.csvfile.close()

class CSVReader():
    '''CSV文件读取通用类
    '''
    
    '''构造函数，需要提供文件路径
    '''
    def __init__(self, filePath):
        self.filePath = filePath
        self.csvfile = file(filePath, 'rb')
        #self.reader = csv.reader(self.csvfile)
    
    '''读多行数据
    '''
    def readRows(self):
        return csv.reader(self.csvfile)
        
    '''关闭
    '''
    def close(self):
        self.csvfile.close()

if __name__ == '__main__': 
    pass



