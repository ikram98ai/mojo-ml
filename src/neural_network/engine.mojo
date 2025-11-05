from hashlib.hasher import Hasher
from collections import Set
import math


struct Value(Stringable, Writable, ImplicitlyCopyable, Copyable, Movable, Hashable, EqualityComparable):
    var data: Float32
    var grad: Float32
    var _prev: List[Self]
    var _op: String
    var _backward: fn() escaping -> None

    fn __init__(out self, x:Float32, var child:List[Self]=List[Self](), op:String= ''):
        self.data = x
        self.grad = Float32(0)
        self._prev = child^
        self._op = op
        fn inline_backward() escaping -> None:
            pass
        self._backward = inline_backward


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
        self._backward = existing._backward

       
    fn __moveinit__(out self, deinit existing: Self):
        self. data = existing.data
        self. grad = existing.grad
        self._prev = existing._prev^
        self._op = existing._op       
        self._backward = existing._backward

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

    fn __add__(mut self, mut other:Self) -> Self:
        out = Self(self.data + other.data, [self.copy(), other.copy()], "+")

        fn _backward():
            self.grad  += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    fn __mul__(mut self, mut other:Self) -> Self:
        out = Self(self.data*other.data,[self.copy(), other.copy()], "*")
      
        fn _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    fn tanh(mut self) -> Self:
        x = self.data
        t = (math.exp(2*x)-1)/ (math.exp(2*x)+1)
        out = Self(t, [self.copy()], 'tanh')
       
        fn _backward():
            self.grad += (1-t**2) * out.grad

        out._backward = _backward
        return out.copy()
    
    fn backward(mut self):
        # topological order all of the children in the graph
        visited = Set[Self]()
        topo = List[Self]()
        @parameter
        fn build_topo(v:Self):
            if v not in visited:
                visited.add(v)
                for ref child in v._prev:
                    build_topo(child)
                topo.append(v.copy())

        self.grad = Float32(1)
        build_topo(self)
        for v in topo[::-1]:
            v._backward()

def main():
    v1 = Value(8)
    v2 = Value(3)
    v3 = v1 + v2
    v4 = v3.tanh()
    v4.backward()
    print(v4)