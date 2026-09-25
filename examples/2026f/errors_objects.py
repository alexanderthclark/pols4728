class PredictionErrors:
    def __init__(self, errors):
        self.errors = errors

    def sum_of_squares(self):
        return sum(error ** 2 for error in self.errors)

sample = PredictionErrors([1, -2, 3])
print(sample.sum_of_squares())  # 14
