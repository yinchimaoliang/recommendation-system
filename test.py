TEST_RESULT = './result/test_result.txt'
TEST_VAL = './data/test_result.txt'
import math

if __name__ == '__main__':
    test_result = []
    test_val = []
    diff = []
    with open(TEST_RESULT,'r') as f:

        while True:
            line = f.readline()

            if not line or line == '\n':
                break
            id = line.split('|')
            item_num = 6
            for i in range(item_num):
                line = f.readline()
                item_id, score = line.split(" ")[:2]
                score = int(score)
                test_result.append(score)


    with open(TEST_VAL,'r') as f:
        while True:
            line = f.readline()

            if not line or line == '\n':
                break
            id, item_num = line.split('|')
            item_num = int(item_num[:-1])
            for i in range(item_num):
                line = f.readline()
                item_id, score = line.split("  ")[:2]
                score = int(score)
                test_val.append(score)


    for i in range(len(test_result)):
        diff.append(abs(test_result[i] - test_val[i]) ** 2)
    print(test_result)
    print(test_val)
    print(math.sqrt(sum(diff)) / len(diff))