class Tree:
    def __init__(self):
        self.isLeaf  = False

        """
        ATTENTION!!!
        If it is not a leaf, the self.nodeVal is threshold
        otherwise, self.nodeVal is labels
        """
        #-----------------------------------------------
        self.nodeVal = None
        #------------------------------------------------

        self.selFeature = None # selected feature

        self.counter = {} #how many samples at this left

        self.left    = None
        self.right   = None
