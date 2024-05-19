from sklearn.metrics import confusion_matrix


class Metrics:

    def __init__(self, true_values, predictions):
        matrix = confusion_matrix(true_values, predictions)
        if matrix.shape == (2, 2):
            self.tp = matrix[1][1]
            self.tn = matrix[0][0]
            self.fp = matrix[0][1]
            self.fn = matrix[1][0]
        else:
            self.tp = self.tn = self.fp = self.fn = 0

    @property
    def accuracy(self):
        if self.tp == self.tn == self.fp == self.fn == 0:
            return 0
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    @property
    def precision(self):
        if self.tp == self.tn == self.fp == self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    @property
    def iou(self):
        if self.tp == self.tn == self.fp == self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fp + self.fn)

    @property
    def recall(self):
        if self.tp == self.tn == self.fp == self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    @property
    def f1_score(self):
        if self.tp == self.tn == self.fp == self.fn == 0:
            return 0
        precision = self.precision
        recall = self.recall
        return (2 * precision * recall) / (precision + recall)
