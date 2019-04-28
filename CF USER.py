import numpy as np
from scipy import sparse
import time


ITEM_PATH = './data/itemAttribute.txt'
TRAIN_PATH = "./data/train.txt"
TEST_PATH = "./data/test.txt"
RESULT_PATH = "./result/CF_USER_result.txt"
START = 18000
END = 20000
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
        mat_1 = np.array(user_sims)
        mat_2 = np.array(self.rating_matrix[:,item_no].toarray()).flatten()
        set = mat_2.nonzero()[0]
        mat_3 = np.array(self.rating_aves)
        mat_result = np.dot(mat_1[set],(mat_2[set] - mat_3[set]))
        return mat_result / np.sum(mat_1[set]) + self.rating_aves[user_no]


    #计算某用户对所有用户的相似度
    def computeUserSim(self,id):
        #存放用户相似度
        user_sims = []
        user_no = self.user_dic[id]



        set = [self.item_dic[self.users[user_no].items[i][0]] for i in range(self.users[user_no].item_num)]
        mat_1 = self.rating_matrix[user_no, set]
        mat_2 = self.rating_matrix[:, set]
        mat_result = mat_1.dot(mat_2.T)
        # print(mat_result)
        # print(set)
        for j in range(self.user_num):
            # print(sum(self.rating_matrix[user_no,set] * self.rating_matrix[j,set]))
            num = mat_result[0,j]
            denom = self.rating_aves[user_no] * self.users[user_no].item_num * self.rating_aves[j] * self.users[j].item_num
            cos = num / denom
            sim = 0.5 + 0.5 * cos
            user_sims.append(sim)

        return user_sims



    def predict(self):
        for i in range(START,END):
            start = time.time()
            print('start predict' + str(i))
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
            end = time.time()
            print('finish predict %d,using %f seconds' % (i,end - start))
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