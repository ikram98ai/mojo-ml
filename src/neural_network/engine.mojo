from python import Python, PythonObject
from collections import Set
from hashlib.hasher import Hasher


struct Value(Stringable, Writable, Copyable, Movable, Hashable, EqualityComparable):
    var data: Float32
    var grad: Float32
    var _prev: List[Self]
    var _op: String
    var label: String

    fn __init__(out self, x:Float32, var child:List[Self]=List[Self](), op:String= '', label:String = ""):
        self.data = x
        self.grad = Float32(0)
        self._prev = child^
        self._op = op
        self.label = label

    fn __str__(self)-> String:
            return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        self_label = String(self.label,"(d:",self.data,"|g:",self.grad,")")
        if len(self._prev) == 1 :
            writer.write(self._prev[0].label,"(",self._prev[0].data,")",self._op," = ", self_label)
        elif len(self._prev) == 2 :
            writer.write(self._prev[0].label,"(",self._prev[0].data,")",self._op,self._prev[1].label,"(", self._prev[1].data,")"," = ", self_label)
        else:
            writer.write(self_label)


    fn __copyinit__(out self, existing: Self):
        self. data = existing.data
        self. grad = existing.grad
        self._prev = existing._prev.copy()
        self._op = existing._op
        self.label =existing.label

    fn __moveinit__(out self, deinit existing: Self):
        self. data = existing.data
        self. grad = existing.grad
        self._prev = existing._prev^
        self._op = existing._op
        self.label =existing.label

    fn __eq__(self, other: Self) -> Bool:
        is_equal = self.data == other.data and self.grad == other.grad and self._op == other._op 
        if len(self._prev) >=1:
            is_equal = is_equal and self._prev[0] == other._prev[0] 
        if len(self._prev) ==2:
            is_equal = is_equal and self._prev[1] == other._prev[1]
        return is_equal

    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self.data)

    fn __add__(self, other:Self) -> Self:
        return Self(self.data+other.data, [self.copy(), other.copy()], "+")
 
    fn __sub__(self, other:Self) -> Self:
        return Self(self.data-other.data, [self.copy(), other.copy()], "-")

    fn __mul__(self, other:Self) -> Self:
        return Self(self.data*other.data, [self.copy(), other.copy()], "*")
    
    fn __truediv__(self, other:Self)raises -> Self:
        if other.data == 0:
            raise Error("The denominator should not be zero.")
        return Self(self.data/other.data, [self.copy(), other.copy()], "/")
    


def main():
    v1 = Value(8,label='v1')
    v2 = Value(3,label='v2')
    v3 = v2/v1
    v3.label = "v3"
    
    print(v3)