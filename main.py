import numpy as np


CF_PATH = './result/CF_USER_result.txt'
SVD_PATH = './result/SVD_result.txt'
RESULT_PATH = './result/result.txt'
TEST_PATH = "./data/test.txt"




#User数据类型
class User():
    def __init__(self,id,item_num):
        self.id = id
        self.items = []
        self.item_num = item_num

    def setItems(self,item):
        self.items.append(item)

    def getAverage(self):
        t = np.array(self.items,dtype = 'uint')
        return np.mean(t[:,1])

#盛放result
class Result():
    def __init__(self,id,num):
        self.id = id
        self.items = []
        self.num = num
    def setItems(self,item):
        self.items.append(item)


class Main():
    def __init__(self):
        self.test = []
        self.CF_results = []
        self.SVD_results = []
        self.user_dic = {}


    #获取两个文件的预测结果
    def getData(self):
        with open(TEST_PATH,'r') as f:
            i = 0
            while True:
                line = f.readline()
                if not line or line == '\n':
                    break

                id,item_num = line.split('|')
                item_num = int(item_num[:-1])
                user = User(id,item_num)
                for i in range(item_num):
                    line = f.readline()
                    item_id = line[:-1]
                    user.setItems([item_id])
                self.test.append(user)
                self.user_dic[id] = i
                i += 1
        #获取CF结果
        with open(CF_PATH,'r') as f:
            while True:
                line = f.readline()
                if not line or line == '\n':
                    break

                id = line[:-1]
                num = self.test[self.user_dic[id]].item_num
                print(id)
                result = Result(id,num)
                for i in range(num):
                    line = f.readline()
                    item_id,score = line.split(':')
                    score = round(float(score[:-1]))

                    result.setItems([item_id,score])

                self.CF_results.append(result)
        #获取SVD结果
        with open(SVD_PATH, 'r') as f:
            while True:
                line = f.readline()
                if not line or line == '\n':
                    break

                id = line[:-1]
                num = self.test[self.user_dic[id]].item_num
                result = Result(id,num)
                for i in range(num):
                    line = f.readline()
                    item_id, score = line.split(':')
                    score = round(float(score[:-1]))

                    result.setItems([item_id, score])

                self.SVD_results.append(result)

    #将结果写入目标文件
    def setData(self):
        with open(RESULT_PATH,'w') as f:
            for i in range(len(self.CF_results)):
                f.write(self.CF_results[i].id)
                f.write('\n')

                for j in range(self.CF_results[i].num):
                    item_id = self.CF_results[i].items[j][0]
                    score = self.CF_results[i].items[j][1]

                    #如果某条结果协同过滤结果有误，则采用SVD结果
                    if score < 0:
                        score = self.SVD_results[i].items[j][1]
                    if score > 100:
                        score = 100
                    f.write(item_id)
                    f.write(' ')
                    f.write(str(score))
                    f.write('\n')


    def mainMethod(self):
        self.getData()
        self.setData()



if __name__ == '__main__':
    t = Main()
    t.mainMethod()
