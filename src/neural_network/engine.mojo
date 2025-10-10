from python import Python, PythonObject
from hashlib.hasher import Hasher
from collections import Set
import math

struct Value(Stringable, Writable, Copyable, Movable, Hashable, EqualityComparable):
    var data: Float32
    var grad: Float32
    var _prev: List[Self]
    var _op: String

    fn __init__(out self, x:Float32, var child:List[Self]=List[Self](), op:String= ''):
        self.data = x
        self.grad = Float32(0)
        self._prev = child^
        self._op = op

    fn __str__(self)-> String:
            return String.write(self)

    fn write_to(self, mut writer: Some[Writer]):
        self_label = String("(d:",self.data,"|g:",self.grad,")")
        if len(self._prev) == 1 :
            writer.write(self._prev[0].data,self._op," = ", self_label)
        elif len(self._prev) == 2 :
            writer.write(self._prev[0].data,self._op, self._prev[1].data," = ", self_label)
        else:
            writer.write(self_label)

    fn __copyinit__(out self, existing: Self):
        self. data = existing.data
        self. grad = existing.grad
        self._prev = existing._prev.copy()
        self._op = existing._op

    fn __moveinit__(out self, deinit existing: Self):
        self. data = existing.data
        self. grad = existing.grad
        self._prev = existing._prev^
        self._op = existing._op
    
    fn __eq__(self, other: Self) -> Bool:
        is_equal = self.data == other.data and self.grad == other.grad and self._op == other._op 
        # if len(self._prev) >=1:
        #     is_equal = is_equal and self._prev[0] == other._prev[0] 
        # if len(self._prev) ==2:
        #     is_equal = is_equal and self._prev[1] == other._prev[1]
        return is_equal

    fn __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self.data)

#################################################### Value operaitons ####################################################

    fn __add__(self, other:Self) -> Self:
        out =self.data+other.data
        return Self(out, [self.copy(), other.copy()], "+")

    fn __mul__(self, other:Self) -> Self:
        out = self.data*other.data
        return Self(out,[self.copy(), other.copy()], "*")

    fn tanh(self) -> Self:
        x = self.data
        t = (math.exp(2*x)-1)/ (math.exp(2*x)+1)
        return Self(t, [self.copy()], 'tanh')
    
    fn backward(mut self):
        # topological order all of the children in the graph
        visited = Set[Self]()
        topo = List[Self]()
        @parameter
        fn build_topo(ref v:Self):
            if v not in visited:
                visited.add(v)
                for ref child in v._prev:
                    build_topo(child)
                topo.append(v)

        self.grad = Float32(1)
        build_topo(self)

        for ref v in topo[::-1]:
            if v._op == "+":
                v._prev[0].grad += v.grad
                v._prev[1].grad += v.grad
            elif v._op == "*":
                v._prev[0].grad += v._prev[1].data * v.grad
                v._prev[1].grad += v._prev[0].data * v.grad
            elif v._op == "tanh":
                v._prev[0].grad += (1-v._prev[0].data**2) * v.grad


def main():
    v1 = Value(8)
    v2 = Value(3)
    v3 = v1 + v2
    v4 = v3.tanh()
    v4.backward()
    print(v4)