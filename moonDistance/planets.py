class Planet:
    def __init__(self, x, y, r):
        # x and y are the coordinates of the center of the circle
        self.x = x
        self.y = y
        self.r = r
        self.fileName = ""
        self.type = ""
    
    # create getter methods for the x, y, and r attributes
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getR(self):
        return self.r
    
    def setFileName(self, fileName):
        self.fileName = fileName
    
    def setType(self, type):
        self.type = type