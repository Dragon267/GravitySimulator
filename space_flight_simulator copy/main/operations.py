import math

class operations:
    def minus(self, vec1, vec2):
        ans = []
        for i in range(len(vec1)):
            ans.append(vec1[i] + vec2[i])
        return ans
    
    def plus(self, vec1, vec2):
        ans = []
        for i in range(len(vec1)):
            ans.append(vec1[i] + vec2[i])
        return ans
    
    def multi(self, vec, n):
        ans = []
        for i in range(len(vec)):
            ans.append(vec[i] * n)
        return ans
    
    def divide(self, vec, n):
        ans = []
        for i in range(len(vec)):
            ans.append(vec[i] / n)
        return ans
    
    def magnitude(self, vec):
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2)