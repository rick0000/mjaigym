from mjaigym.board.function.pai import Pai


class Mentsu:
    def __init__(self, pais, type, visibility):
        self.pais = sorted(pais, key=Pai.sort)
        self.type = type
        self.visibility = visibility
        self._key = f"{self.type}{self.pais[0]}{self.visibility}"

    def __eq__(self, other):
        if not isinstance(other, Mentsu):
            return NotImplemented
        return self._key == other._key

    def __lt__(self, other):
        if not isinstance(other, Mentsu):
            return NotImplemented
        return self._key < other._key

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)
