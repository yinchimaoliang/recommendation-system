#从训练集中切出测试集和训练集
DATA_PATH = './data/train.txt'
TRAIN_DATA_PATH = './data/test_train.txt'
TEST_DATA_PATH  = './data/test_test.txt'
TEST_RESULT_PATH = './data/test_result.txt'
TEST_NUM = 100
THRESHOLD = 10

def getTestData():
    with open(DATA_PATH,'r') as f:
        user_no = 0

        while True:
            line = f.readline()

            if not line or line == '\n':
                break
            id, item_num = line.split('|')
            item_num = int(item_num[:-1])
            if user_no < TEST_NUM and item_num > THRESHOLD:
                with open(TEST_DATA_PATH,'a') as p:
                    p.write('%s|%d\n'%(id,6))
                for i in range(item_num):
                    line = f.readline()
                    item_id, score = line.split("  ")[:2]
                    score = int(score)
                    if i < 6:
                        with open(TEST_DATA_PATH, 'a') as p:
                            p.write(item_id)
                            p.write('\n')
            else:
                for i in range(item_num):
                    line = f.readline()

            user_no += 1

def getTestResult():
    with open(DATA_PATH,'r') as f:
        user_no = 0

        while True:
            line = f.readline()

            if not line or line == '\n':
                break
            id, item_num = line.split('|')
            item_num = int(item_num[:-1])
            if user_no < TEST_NUM and item_num > THRESHOLD:
                with open(TEST_RESULT_PATH,'a') as p:
                    p.write('%s|%d\n'%(id,6))
                for i in range(item_num):
                    line = f.readline()
                    item_id, score = line.split("  ")[:2]
                    score = int(score)
                    if i < 6:
                        with open(TEST_RESULT_PATH, 'a') as p:
                            p.write('%s  %d\n' % (item_id, score))
            else:
                for i in range(item_num):
                    line = f.readline()

            user_no += 1



def getTrainData():
    with open(DATA_PATH,'r') as f:
        user_no = 0

        while True:
            line = f.readline()

            if not line or line == '\n':
                break
            id, item_num = line.split('|')
            item_num = int(item_num[:-1])
            if user_no < TEST_NUM and item_num > THRESHOLD:
                with open(TRAIN_DATA_PATH,'a') as q:
                    q.write('%s|%d\n'%(id,item_num - 6))
                for i in range(item_num):
                    line = f.readline()
                    item_id, score = line.split("  ")[:2]
                    score = int(score)
                    if i >= 6:
                        with open(TRAIN_DATA_PATH, 'a') as q:
                            q.write('%s  %d\n' % (item_id,score))
            else:
                with open(TRAIN_DATA_PATH, 'a') as q:
                    q.write('%s|%d\n' % (id, item_num))
                    for i in range(item_num):
                        line = f.readline()
                        item_id, score = line.split("  ")[:2]
                        score = int(score)
                        q.write('%s  %d\n' % (item_id, score))

            user_no += 1



if __name__ == '__main__':
    getTrainData()
    getTestData()
    getTestResult()