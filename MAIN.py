from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd



ITEM_PATH = './data/itemAttribute.txt'
USER_PATH = "./data/train.txt"


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
        self.ratings = []
        #Item id到self.items的映射
        self.item_dic = {}
        #user id到self.users的映射
        self.user_dic = {}



    def getData(self):


        #获取items
        with open(ITEM_PATH,'r') as f:
            item_no = 0
            while True:
                line = f.readline()
                if not line:
                    break
                id,attr1,attr2 = line.split('|')
                attr2 = attr2[:-1]
                if attr1 == 'None':
                    attr1 = None

                if attr2 == 'None':
                    attr2 = None
                item = Item(id)
                item.setAttr(attr1,attr2)
                self.items.append(item)
                self.item_dic[id] = item_no
                item_no += 1

        self.item_num = len(self.items)

        #获取users
        with open(USER_PATH,'r') as f:
            user_no = 0
            while True:
                line = f.readline()
                user_no += 1
                if not line or line == '\n':
                    break
                id,item_num = line.split('|')
                item_num = int(item_num[:-1])
                user = User(id,item_num)
                for i in range(item_num):
                    line = f.readline()
                    item_id,score = line.split("  ")[:2]
                    score = int(score)
                    user.setItems([item_id,score])
                    self.ratings.append([id,item_id,score / 20])

                self.user_dic[id] = item_num


                self.users.append(user)
        self.user_num = len(self.users)


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
        trainset, testset = train_test_split(self.data, test_size=.25)
        model = SVD(n_factors=1000)
        model.fit(trainset)
        a_user = "0"
        a_product = "507696"
        print(model.predict(a_user, a_product))

    def mainMethod(self):
        self.getData()
        self.mySVD()
        # for i in range(100):
        #     print(self.item_dic[self.items[i].id])
        #     print(self.items[i].id,self.items[i].attr1,self.items[i].attr2)
        # print(self.items)
        # self.myCF()
        # print(self.ratings[:100])


if __name__ == '__main__':
    t = Main()
    t.mainMethod()