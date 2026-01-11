#assumes X,A,y are pytorch tensors

class GraphData:
    def __init__(self, X, A, y, requires_grad=False):
        self.X = X
        self.A = A
        self.y = y

        if requires_grad:
            self.X.requires_grad_(True)
            self.y.requires_grad_(True)
