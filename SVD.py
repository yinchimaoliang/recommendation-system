import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd
from scipy import sparse
import time


ITEM_PATH = './data/itemAttribute.txt'
TRAIN_PATH = "./data/train.txt"
TEST_PATH = "./data/test.txt"
RESULT_PATH = "./result/SVD_result.txt"
SVD_PARAMETER = 10


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

                # print(id)
                self.users.append(user)
        self.user_num = len(self.users)
        self.item_num = len(self.items)
        self.rating_matrix = sparse.dok_matrix((self.user_num, self.item_num))
        # print(self.item_dic['507696'])
        for i in range(self.user_num):
            for j in range(self.users[i].item_num):
                self.rating_matrix[self.user_dic[self.users[i].id],self.item_dic[self.users[i].items[j][0]]] = self.users[i].items[j][1]

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
        # for i in self.test:
        #     print(i.id,i.items)
        print('finish getData')



    def mySVD(self):
        self.reader = Reader(rating_scale = (1,5))
        self.data = Dataset.load_from_df(pd.DataFrame(self.ratings),self.reader)
        print(self.data)
        trainset, testset = train_test_split(self.data, test_size=.15)
        self.model = SVD(n_factors=SVD_PARAMETER)
        self.model.fit(trainset)
        a_user = "0"
        a_product = "507696"
        print(self.model.predict(a_user, a_product))




    def predict(self):
        for i in range(self.test_num):
            with open(RESULT_PATH,'a') as f:
                f.write(self.test[i].id)
                f.write('\n')
                for j in range(len(self.test[i].items)):
                    self.test[i].items[j].append(self.model.predict(self.test[i].id,self.test[i].items[j][0])[3] * 20)
                    f.write(self.test[i].items[j][0])
                    f.write(':')
                    f.write(str(self.test[i].items[j][1]))
                    f.write('\n')

    def mainMethod(self):
        self.getData()
        start = time.clock()
        self.mySVD()
        self.predict()
        elapsed = (time.clock() - start)
        print(elapsed)


if __name__ == '__main__':
    t = Main()
    t.mainMethod()