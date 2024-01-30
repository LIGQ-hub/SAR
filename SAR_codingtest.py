import numpy as np
import os
from PIL import Image

class SarDataProcess:
    def __init__(self, num, sarDataFolder,
                 prefix, suffix, row, column):
        """

        :param num: 处理的图片数量
        :param sarDataFolder:  SAR图像数据存放的文件夹地址
        :param prefix: SAR图像数据保存的文件名前缀
        :param suffix: SAR图像数据保存的文件名后缀
        :param row: SAR图像数据保存的文件处理后矩阵的行数
        :param column: SAR图像数据保存的文件处理后矩阵的列数
        """
        self.num = num
        self.prefix = prefix
        self.suffix = suffix
        self.sarDataFolder = sarDataFolder
        self.row = row
        self.column = column
    def getMeanValueOfdata(self):
        """
        获得数据组的平均幅度
        :return:
        """
        for i in range(self.num):
            currentdataPath = os.path.join(self.sarDataFolder, self.prefix + str(i) + self.suffix)    # 假设文件按规律命名
            currentdata = self.loadData(currentdataPath)   # 读取单幅文件
            dataAmplitude = np.abs(currentdata)            # 求幅度
            if i == 0:
                sumValue = dataAmplitude
            else:
                sumValue = np.add(dataAmplitude, sumValue)     # 把每幅文件求幅度相加
        meanValue = sumValue / self.num                    # 求平均幅度
        np.save("meanValue", meanValue)
        return meanValue

    def getMeanAmplitudeJPG(self):
        """
        获得平均幅度的JPG
        """
        meanValue = self.getMeanValueOfdata()
        self.saveGenerateJPG(meanValue, 'meanAmplitude.jpg')

    def getStandardDeviation(self):
        for i in range(self.num):
            currentdataPath = os.path.join(self.sarDataFolder, self.prefix + str(i) + self.suffix)
            currentdata = self.loadData(currentdataPath)
            dataSquareAmplitude = np.square(np.abs(currentdata))   # 方差等于平方的均值减均值的平方
            if i == 0:
                stdVia = dataSquareAmplitude
            else:
                stdVia = np.add(dataSquareAmplitude, stdVia)         # 为了减小内存使用，这里的stdVia实际上为dataSquareAmplitude
        # 下面都对stdVia进行了覆盖
        stdVia = stdVia / self.num
        meanValue = np.load('meanValue.npy')     # 由于内存问题，无法同时计算均值，所有要先进行均值运算，保存后调用
        stdVia = np.sqrt(stdVia - np.square(meanValue))              # 计算标准差
        meanValue[np.where(meanValue == 0)] = 0.0001                 # 将meanValue矩阵中所有零值幅值为0.0001
        stdVia = stdVia / meanValue                                  # 计算归一化标准差
        self.saveGenerateJPG(stdVia, 'stdVia.jpg')

    def loadData(self, dataPath):
        """
        读取文件(若雷达数据为二进制保存)
        此处只为了表示怎么读取二进制保存的雷达数据
        在实际中，SLC为int32保存,若SLC数据也为二进制保存,那么使用此方法读取加计算，内存将不够
        :param dataPath: 数据文件地址
        :return: 返回复数矩阵
        """
        with open(dataPath, 'rb') as file:
            data = np.fromfile(file, dtype=np.int32)    # 读取二进制文件转换为int32
        # 按照存储规则将得到实部矩阵和虚部矩阵
        d_re = data[0:len(data):2]  # 取实数部分数据
        d_im = data[1:len(data):2]  # 取虚数部分数据
        # 将实部和虚部合并成复数矩阵
        comp = d_re+d_im*1j
        comp_matrix = np.array(comp).reshape((self.row, self.column))
        return comp_matrix

    def saveGenerateJPG(self, imgData, fineName):
        """
        保存处理后生成的JPG
        :param imgData: 处理后的图像数据
        :param fineName: 保存的图像名称
        :return:
        """
        jpgData = Image.fromarray(imgData)
        jpgData = jpgData.convert('RGB')
        jpgData.save(os.path.join(self.sarDataFolder, fineName))


if __name__ == "__main__":
    sardataProcess = SarDataProcess()
    # 若需要计算归一化标准差，请先计算均值
    sardataProcess.getMeanAmplitudeJPG()
    sardataProcess.getStandardDeviation()