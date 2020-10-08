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
    
    def get_point(self):
        return self.point_info.points

    def get_yakus(self):
        return [y[0] for y in self.point_info.yakus]

    def valid(self):
        try:
            return len([y for y in self.point_info.yakus if y[0] not in ["dora", "akadora", "uradora"]]) > 0
        except:
            import pdb; pdb.set_trace(); import time; time.sleep(1)
            print(self.point_info)
            False

    def distance(self):
        return sum([d for d in self.diff if d > 0])

    def __repr__(self):
        return f"{self.combination}, {self.point_info}, {self.diff}"

