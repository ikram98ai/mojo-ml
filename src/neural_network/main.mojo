from python import Python
from layout import Layout
import math
import random


def load_data(file_path:String)-> (List[List[Float32]], List[Float32]):
    pl = Python.import_module("polars")
    df = pl.read_csv(file_path)
    feature_df = df.drop('target')
    var features: List[List[Float32]] = [[ Float32(x) for x in row] for row in feature_df.iter_rows()]
    var labels: List[Float32] = [Float32(y) for y in df['target']]
    return features^, labels^



struct Linear(Copyable, Movable):
    var w: List[List[Float32]]
    var b: List[Float32]
    var act: Bool
    fn __init__(out self, input:Int, output:Int, act:Bool = True):
        """
        shape(3,2).
    
            a   b   c       k  l        k*a + m*b + o*c     l*a + n*b + p*c         u   v
            d   e   f   *   m  n    =   k*d + m*b + o*c     l*a + n*b + p*c     =   w   x
            g   h   i       o  p        k*g + m*b + o*c     l*a + n*b + p*c         y   z
        """
        self.act=act
        random.seed(42)
        self.w = List[List[Float32]]()
        self.b = List[Float32]()

        for _ in range(input):
            var layer_w = List[Float32]()
            for _ in range(output):
                layer_w.append(Float32(random.randn_float64()))
            self.w.append(layer_w.copy())
            self.b.append(1.0)

    fn __call__(self, x:List[List[Float32]]) raises -> List[List[Float32]]:
        m1 = len(x)
        n1 = len(x[0])
        m2 = len(self.w)
        n2 = len(self.w[0])
        if n1 != m2:
            raise Error("The columns of 1st matrix must be equal to the rows of 2nd matrix.")

        c = List[List[Float32]]()
        for i in range(m1):
            row = List[Float32]()
            for j in range(n2):
                z:Float32 = 0.0
                for k in range(n1):
                    z += x[i][k] * self.w[k][j]
                z+= self.b[i]
                if self.act:
                    row.append(self._relu(z))
                else:
                    row.append(z)
            c.append(row.copy())
        return c^

    fn _sigmoid(self, z:Float32)->Float32:
        return 1/(1 + math.exp(-z))   

    fn _relu(self, z:Float32)->Float32:
        return z if z>0 else 0 

fn softmax(z:List[List[Float32]]) -> List[Float32]:
    output = List[Float32]()
    for i in range(len(z)):
        z_sum = Float32(0)
        for j in range(len(z[0])):
            z_sum += math.exp(z[i][j])
        max = Float32(0)
        for j in range(len(z[0])):
            new = math.exp(z[i][j])/z_sum
            if new > max:
                max = new

        output.append(max)
    return output^


def main():
    data = load_data("./././data/heart_disease.csv")
    x = data[0].copy()
    
    var layers: List[Linear] = [ 
        Linear(13,8),
        Linear(8,4,act=False),
        # Linear(4,2),
        # Linear(2,1),
    ]

    var a:List[List[Float32]] = x^
    print(String("Input shape: ({},{})").format(len(a),len(a[0])))
    for i,l in enumerate(layers):
        a = l(a).copy()
        print(String("shape of layer_{}: ({},{})").format(i,len(a),len(a[0])))
    pred = softmax(a)
    for i in range(10):
        print(pred[i])
        
    # for i in range(10):
    #     for j in range(len(a[0])):
    #         print(a[i][j],end="    ")
    #     print()