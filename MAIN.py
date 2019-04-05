

ITEM_PATH = './data/itemAttribute.txt'
USER_PATH = "./data/train.txt"


class Item():
    def __init__(self,id):
        self.id = id


    def setAttr(self,attr1,attr2):
        self.attr1 = attr1
        self.attr2 = attr2


class User():
    def __init__(self,id):
        self.id = id
        self.items = []

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
            line_no = 0
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
                self.item_dic[id] = line_no
                line_no += 1









    def mainMethod(self):
        self.getData()
        for i in range(100):
            print(self.item_dic[self.items[i].id])
            print(self.items[i].id,self.items[i].attr1,self.items[i].attr2)
        # print(self.items)




if __name__ == '__main__':
    t = Main()
    t.mainMethod()