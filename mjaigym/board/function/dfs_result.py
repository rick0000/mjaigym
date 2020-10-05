from enum import Enum

class DfsResultType(Enum):
    Normal=0
    Chitoitsu=1
    Kokushimuso=2

class DfsResult:
    def __init__(self, result_type, combination, point_info, diff):
        self.result_type = result_type
        self.combination = combination
        self.point_info = point_info
        self.diff = diff
        self.dahaiable_ids = [i for (i,num) in enumerate(diff) if num < 0]

    def is_dahaiable(self, dahai_id):
        return dahai_id in self.dahaiable_ids

    def __repr__(self):
        return f"{self.combination}, {self.point_info}, {self.diff}"