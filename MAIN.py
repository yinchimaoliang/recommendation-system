from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd
from scipy import sparse
from math import sqrt
from sklearn.neighbors import NearestNeighbors


ITEM_PATH = './data/itemAttribute.txt'
USER_PATH = "./data/temp.txt"
TEST_PATH = "./data/test.txt"
RESULT_PATH = "./result.txt"
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
        #Item id到self.items的映射
        self.item_dic = {}
        #user id到self.users的映射
        self.user_dic = {}



    def getData(self):


        #获取items
        # with open(ITEM_PATH,'r') as f:
        #     # item_no = 0
        #     while True:
        #         line = f.readline()
        #         if not line:
        #             break
        #         id,attr1,attr2 = line.split('|')
        #         attr2 = attr2[:-1]
        #         if attr1 == 'None':
        #             attr1 = None
        #
        #         if attr2 == 'None':
        #             attr2 = None
        #         item = Item(id)
        #         item.setAttr(attr1,attr2)
        #         self.items.append(item)
        #         # self.item_dic[id] = item_no
        #         # item_no += 1
        #
        # self.item_num = len(self.items)

        #获取users
        with open(USER_PATH,'r') as f:
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


    def myCF(self):
        self.train_data_matrix = np.zeros([self.user_num,self.item_num])
        for i in range(self.user_num):
            for j in range(len(self.users[i].items)):
                col = self.item_dic[self.users[i].items[j][0]]
                self.train_data_matrix[i][col] = self.users[i].items[j][1]

        print(self.train_data_matrix.size)

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



    def myPearson(self,n1,n2):
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        items_1 = np.array(self.users[n2].items)
        items_2 = np.array(self.users[n2].items)
        for key in items_1:
            if key[0] in items_2[:,0]:
                n += 1
                x = self.rating_matrix[n1,self.item_dic[key[0]]]
                y = self.rating_matrix[n2,self.item_dic[key[0]]]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0

            # 皮尔逊相关系数计算公式
        denominator = sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n)
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / n) / denominator


    def computeUserSim(self,user_id):
        n1 = self.user_dic[user_id]
        dists = []
        for n2 in range(self.user_num):
            if n2 != n1:
                dist = self.myPearson(n1,n2)
                dists.append(dist)

        dists.sort()
        print(dists)
        return dists




    def myCosSim(self,n1,n2):
        vec1 = [0 for i in range(self.item_num)]
        vec2 = [0 for i in range(self.item_num)]

        

    def predict(self):
        with open(RESULT_PATH,'w') as f:
            for i in range(self.test_num):
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
        self.computeUserSim('0')
        # self.mySVD()
        # self.predict()
        # for i in range(100):
        #     print(self.item_dic[self.items[i].id])
        #     print(self.items[i].id,self.items[i].attr1,self.items[i].attr2)
        # print(self.items)
        # self.myCF()
        # print(self.ratings[:100])


if __name__ == '__main__':
    t = Main()
    t.mainMethod()