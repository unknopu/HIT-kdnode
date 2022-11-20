import time
import numpy as np
import pandas as pd

def computeDist(pt1, pt2):
    vt1 = np.array(list(map(float, pt1)))
    vt2 = np.array(list(map(float, pt2)))
    return np.sqrt(np.sum(np.square(vt1 - vt2)))

class KD_node:
    def __init__(self, point=None, split=None, left=None, right=None):
        self.point = point 
        self.split = split
        self.left = left 
        self.right = right 

class MAIN_FN:
    def __init__(self ):
        self.Cap = [600]
        self.KDNode = KD_node()
        self.Root = KD_node()
        self.Train = list(pd.read_csv("train.csv").to_records())
        self.Test = list(pd.read_csv("test.csv").to_records())
        self.Smurf, self.Normal = 0, 0
        self.TrainTime, self.EndTime = 0, 0
        self.RightRatio = 0
        self._initVariable()

    def _initVariable(self):
        for d in self.Test:
            if d[-1]:
                self.Smurf += 1
        self.Normal = len(self.Test) - self.Smurf

    def my_test(self, TRAINCOUNT):
        _startTrain = time.time()
        root = self._doKDTree(self.KDNode, self.Train[: TRAINCOUNT])
        self.TrainTime = time.time() - _startTrain
        right, err = 0, 0
        _startTime = time.time()
        for i in range(len(self.Train)):
            Point = self._findThePoint(root, self.Test[i])
            _INCREASE = self._isNormal(Point)
            if _INCREASE and (abs(self.Test[i][-1]) < 1e-7):
                right += _INCREASE
            elif not _INCREASE:
                right += _INCREASE
            else:
                err += 1
        self.EndTime = time.time() - _startTime
        self.RightRatio = 100 * (right / (right + err))
    
    def _doKDTree(self, root, data):
        if len(data):
            dimension = len(data[0]) - 1 
            _MAX, split = 0, 0
            for i in range(1, dimension):
                _dl = [t[i] for t in data]
                val = self._variance(_dl)
                if val > _MAX:
                    _MAX, split = val, i
            data.sort(key=lambda t: t[split])
            x = len(data) // 2
            x1 = x // 2
            root = KD_node(data[x], split)
            root.left = self._doKDTree(root.left, data[0:x1])
            root.right = self._doKDTree(root.right, data[(x1 + 1):len(data)])
            return root
    
    def Do(self):
        for i in self.Cap:
            print("现在开始运行, 训练集大小为: " + str(i))
            self.my_test(i)
            print("测试的Smurf数量是: " + str(self.Smurf))
            print("测试的Normal数量是: " + str(self.Normal))
            print("总训练时间:" + str(self.TrainTime) + "s")
            print("总测试时间:" + str(self.EndTime) + "s")
    
    def _variance(self, pack):
        a = list(map(float, pack))
        return np.var(np.array(a))
    
    def _isNormal(self, dist_list):
        timer, exceptor = 0, 0
        for i in dist_list:
            flag = abs(i[-1] - 0.0) < 1e-7
            timer += flag
            exceptor += not flag
        return timer > exceptor
    
    def _findThePoint(self, root, query):
        _bdist = computeDist(query, root.point)
        nodeList = []
        swap = root
        _dl = [swap.point, None, None]
        # --------------        
        while swap:
            nodeList.append(swap)
            dd = computeDist(query, swap.point)
            if _bdist > dd:
                _bdist = dd
            if swap.point[swap.split] > query[swap.split]:
                swap = swap.left
            else:
                swap = swap.right
        # --------------        
        for node in nodeList:
            if not _dl[1]:
                _dl[2], _dl[1] = _dl[1], node.point
            elif not _dl[2]:
                _dl[2] = node.point
            if (abs(query[node.split] - node.point[node.split])) < _bdist:
                swap = node.left
                if query[node.split] < node.point[node.split]:
                    swap = node.right
                if swap:
                    nodeList.append(swap)
                    curDist = computeDist(query, swap.point)
                    _dl[2] = swap.point
                    if not _dl[1] or curDist < computeDist(_dl[1], query):
                        _dl[2], _dl[1] = _dl[1], swap.point
                    elif _bdist > curDist:
                        _bdist = curDist
                        _dl[2], _dl[1] = _dl[1], _dl[0]
                        _dl[0] = swap.point
        return _dl

if __name__ == "__main__":
    fn = MAIN_FN()
    fn.Do()