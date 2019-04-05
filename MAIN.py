

ITEM_PATH = './data/itemAttribute.txt'
USER_PATH = "./data/train.txt"


class Item():
    def __init__(self,id):
        self.id = id


    def setAttr(self,attr1,attr2):
        self.attr1 = attr1
        self.attr2 = attr2


class User():
    def __init__(self,id,item_num):
        self.id = id
        self.items = []
        self.item_num = item_num

    def setItems(self,item):
        self.items.append(item)



class Main():


    def __init__(self):
        self.items = []
        self.users = []
        self.item_dic = {}
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


                self.user_dic[id] = item_num


                self.users.append(user)








    def mainMethod(self):
        self.getData()
        # for i in range(100):
        #     print(self.item_dic[self.items[i].id])
        #     print(self.items[i].id,self.items[i].attr1,self.items[i].attr2)
        # print(self.items)




if __name__ == '__main__':
    t = Main()
    t.mainMethod()