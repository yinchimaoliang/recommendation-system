import numpy as np
from scipy import sparse
import time


ITEM_PATH = './data/itemAttribute.txt'
TRAIN_PATH = "./data/train.txt"
TEST_PATH = "./data/test.txt"
RESULT_PATH = "./CF_USER_result.txt"
START = 206
END = 300
SVD_PARAMETER = 2000


#Item数据类型
class Item():
    def __init__(self,id):
        self.id = id


    def setAttr(self,attr1,attr2):
        self.attr1 = attr1
        self.attr2 = attr2

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


class Main():


    def __init__(self):
        #所有的Item项
        self.items = []
        #所有的User项
        self.users = []
        #评分数据
        self.ratings = []
        #测试数据集
        self.test = []
        #用户平均评分
        self.rating_aves = []
        #Item id到self.items的映射
        self.item_dic = {}
        #user id到self.users的映射
        self.user_dic = {}



    def getData(self):

        #获取users
        with open(TRAIN_PATH,'r') as f:
            user_no = 0
            item_no = 0
            while True:
                line = f.readline()

                if not line or line == '\n':
                    break
                id,item_num = line.split('|')
                item_num = int(item_num[:-1])
                user = User(id,item_num)
                for i in range(item_num):
                    line = f.readline()
                    item_id,score = line.split("  ")[:2]
                    score = int(score)
                    if score == 0:
                        score = 1
                    user.setItems([item_id,score])
                    self.ratings.append([id,item_id,score / 20])
                    if item_id not in self.item_dic:
                        self.item_dic[item_id] = item_no
                        item_no += 1
                        self.items.append(Item(item_id))
                self.user_dic[id] = user_no
                user_no += 1
                self.users.append(user)

        #user数量
        self.user_num = len(self.users)
        #item数量
        self.item_num = len(self.items)

        #打分矩阵
        self.rating_matrix = sparse.dok_matrix((self.user_num, self.item_num))
        for i in range(self.user_num):
            for j in range(self.users[i].item_num):
                self.rating_matrix[self.user_dic[self.users[i].id],self.item_dic[self.users[i].items[j][0]]] = self.users[i].items[j][1]
        #计算每个用户的平均打分
        for i in range(self.user_num):
            self.rating_aves.append(self.users[i].getAverage())


        #获取测试数据
        with open(TEST_PATH,'r') as f:
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

        self.test_num = len(self.test)
        print('finish getData')


    #协同过滤算法
    def myCF(self,user_id,item_id,user_sims):
        #转换为列表号
        user_no = self.user_dic[user_id]
        if item_id not in self.item_dic:
            return -1
        item_no = self.item_dic[item_id]
        #分子
        molecule_sum = 0
        #分母
        denominator_sum = 0
        for j in range(self.user_num):
            if self.rating_matrix[j,item_no] == 0:
                continue
            molecule_sum += user_sims[j] * (self.rating_matrix[j,item_no] / 20 - self.rating_aves[j] / 20)
            denominator_sum += user_sims[j]
        return self.rating_aves[user_no] + molecule_sum / denominator_sum * 20


    #计算某用户对所有用户的相似度
    def computeUserSim(self,id):
        #存放用户相似度
        user_sims = []
        user_no = self.user_dic[id]

        print('start computeSim')
        start = time.time()
        set = [self.item_dic[self.users[user_no].items[i][0]] for i in range(self.users[user_no].item_num)]
        mat = self.rating_matrix[user_no, set]
        # print(set)
        for j in range(self.user_num):
            # print(sum(self.rating_matrix[user_no,set] * self.rating_matrix[j,set]))
            num = mat.dot(self.rating_matrix[j,set].T)[0,0]
            denom = self.rating_aves[user_no] * self.users[user_no].item_num * self.rating_aves[j] * self.users[j].item_num
            cos = num / denom
            sim = 0.5 + 0.5 * cos
            user_sims.append(sim)
        # user_sims = matrix_1 * matrix_2
        # print('start computeSim')
        # for i in range(self.user_num):
        #     sim = self.myCosSim(user_no,i)
        #     user_sims.append(sim)
        end = time.time()
        print('finish computeSim,using %d seconds' %(end - start))
        return user_sims



    #计算两用户余弦相似度
    def myCosSim(self,n1,n2):
        num = 0
        for i in range(self.users[n1].item_num):
            num += self.users[n1].items[i][1] * self.rating_matrix[n2,self.item_dic[self.users[n1].items[i][0]]]

        denom = self.rating_aves[n1] * self.users[n1].item_num * self.rating_aves[n2] * self.users[n2].item_num
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

    def predict(self):
        for i in range(START,END):
            user_sims = self.computeUserSim(self.test[i].id)
            with open(RESULT_PATH,'a') as f:

                f.write(self.test[i].id)
                f.write('\n')
                for j in range(len(self.test[i].items)):
                    self.test[i].items[j].append(self.myCF(self.test[i].id,self.test[i].items[j][0],user_sims = user_sims))
                    f.write(self.test[i].items[j][0])
                    f.write(':')
                    f.write(str(self.test[i].items[j][1]))
                    f.write('\n')

    def mainMethod(self):
        #将数据放在内存中
        self.getData()
        start = time.clock()
        self.predict()
        elapsed = (time.clock() - start)
        print(elapsed)

if __name__ == '__main__':
    t = Main()
    t.mainMethod()