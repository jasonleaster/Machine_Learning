"""
Programmer  :   EOF
Date        :   2015.12.10
File        :   knn.py
E-mail      :   jasonleaster@163.com

Here is a implementation of K Nearest Neighbor Algorithm.
The @K in this name means the inputed data have k demention.

Keep in mind that if you want to learn KNN by this demo, you should
known every line in this source file. 

Wikipedia will be you in the learining trip. Enjoy it.
"""
import numpy

"""
You could use any different type of function which help you 
to measure the distance between two sample points. 
For convenient, here we are using Euclidean distance equation.
"""
def euclidean_distance(list1, list2):
    assert isinstance(list1, list)
    assert isinstance(list2, list)

    summer = 0
    for item1, item2 in zip(list1, list2):
        summer += (item1 - item2)**2

    return numpy.sqrt(summer)



class KD_Tree:
    # create a KD tree recusively
    def __init__(self, Mat, depth = 0):
        self._Mat       = numpy.array(Mat)
        self.depth      = depth
        self.SampleDem  = Mat.shape[0] # The demention of samples
        self.SampleNum  = Mat.shape[1] # The number of samples

        """
        Everytime we create a KD tree, we should use a sort helper
        function to get the middle point of the inputed data that
        is the essential point to build a KD-Tree
        """
        self._sort_Mat(depth)

        #The index which is the middle index of sorted samples.
        #This index sperate the sample into left part and right part
        self.seperator = self.SampleNum/2

        self.node = self._Mat[:, self.seperator] #value of this node

        if self.SampleNum == 1:
            self.left = None
            self.right = None
            return

        self.left  = KD_Tree(self._Mat[:,   : self.seperator], depth + 1)

        if self.SampleNum == 2:
            self.right = None
        else:
            self.right = KD_Tree(self._Mat[:, self.seperator+1 :], depth + 1)

    # just a stupid sort function. Forgive me :)
    def _sort_Mat(self, depth):
        dem = depth % self.SampleDem

        for i in range(self.SampleNum):
            for j in range(i+1, self.SampleNum):
                if self._Mat[dem][i] > self._Mat[dem][j]:
                    self._Mat[:, i], self._Mat[:, j] = \
                    numpy.array(self._Mat[:, j]),\
                    numpy.array(self._Mat[:, i])

    #--------- helper function to print the tree using BFS ------
    def __str__(self):
        self.printTree(self)
        return ""

    def printTree(self, tree):
        Q = [tree]
        while len(Q) != 0:
            p = Q.pop(0)
            print str(p.node)
            if p.left != None:
                Q.append(p.left)
            if p.right != None:
                Q.append(p.right)
    # ----------------------------------------------------------


class K_Nearest_Neighbor:
    def __init__(self, Mat, Tag = None):
        self._Mat = numpy.array(Mat)
        if Tag != None: self._Tag = numpy.array(Tag).flatten()

        self.SampleDem = self._Mat.shape[0]
        self.SampleNum = self._Mat.shape[1]

        self.tree = KD_Tree(Mat)

    """
    @distance will calculate the distance between point 
    @P1 and @P2 by using function @func. The default function to 
    figure out the value is @euclidean_distance. You could use 
    another distance equation to implement a distance function.
    """
    def distance(self, P1, P2, func = euclidean_distance):
        return func(P1, P2)

    """
    I have to say sorry. I design this API and change it into a
    better one. Everytime you call this function, you have to pass
    a KD-Tree object into this function as a parameter. I can't 
    eliminate it and use @self.tree.

    Anyway,usage:
        1.Create a @K_Nearest_Neighbor object like this:
            a = K_Nearest_Neighbor(Data)
        2.Then call this function like this:
            a.search(Unkown_Point, a.tree)

    It's not very elegant :)
    """
    def search(self, point, tree = None, depth = 0):
        
        #Don't forget to keep the index(@depth) of point in the 
        #demention of samples. Otherwise, you will get exception
        depth %= self.SampleDem 

        closest = None
        if tree == None:
            return closest

        if point[depth] < tree.node[depth]:    
            closest = self.search(point, tree.left, depth + 1)
        elif point[depth] > tree.node[depth]:
            closest = self.search(point, tree.right, depth + 1)

        if closest == None:
            closest = tree.node
        else:
            D1 = self.distance(list(closest)  , list(point))
            D2 = self.distance(list(tree.node), list(point))
            shortest_dist = D1
            if D2 < D1:
                closest = tree.node
                shortest_dist = D2

            """ !!!Key point.
            Don't forget to check if the other side 
            have more closer point than current side"""
            if abs(tree.node[depth] - point[depth]) < shortest_dist:
                if point[depth] < tree.node[depth]:    
                    may_closest = self.search(point, tree.right, depth + 1)
                elif point[depth] > tree.node[depth]:
                    may_closest = self.search(point, tree.left, depth + 1)

                D1 = self.distance(list(closest)  ,   list(point))
                D2 = self.distance(list(may_closest), list(point))
                shortest_dist = D1
                if D2 < D1:
                    closest = may_closest
                    shortest_dist = D2

        return closest

