class Interval:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    @property
    def mid(self):
        return (self.low + self.high) / 2

    @property
    def rad(self):
        return (self.high - self.low) / 2

    @property
    def wid(self):
        return self.high - self.low

    def __add__(self, other):
        return Interval(self.low + other.low, self.high + other.high)

    def __sub__(self, other):
        return Interval(self.low - other.high, self.high - other.low)

    def __mul__(self, other):
        values = [self.low * other.low, self.low * other.high, self.high * other.low, self.high * other.high]
        return Interval(min(values), max(values))

    def __truediv__(self, other):
        other_reciprocal = Interval(1 / other.high, 1 / other.low)
        return self * other_reciprocal

    def union(self, other):
        return Interval(min(self.low, other.low), max(self.high, other.high))

    def intersect(self, other):
        return Interval(max(self.low, other.low), min(self.high, other.high))

    def inside(self, other):
        return self.low > other.low and self.high < other.high

    def __str__(self):
        return f'[{self.low}, {self.high}]'

    def __repr__(self):
        return str(self)
