from python import Python
import math 

alias simd_width = 8
alias row_type = SIMD[DType.float32, simd_width]

def load_data(file_path:String)-> Tuple[List[row_type], List[Float32]]:
    pl = Python.import_module("polars")
    df = pl.read_csv(file_path)
    features_df = df.select(pl.all().exclude("target"))

    var features = List[row_type]()
    var labels: List[Float32] = [Float32(y) for y in df['target']]
    simd_row = row_type()
    for row in features_df.iter_rows():
        for i in range(simd_width):
            simd_row[i] = Float32(row[i])
        features.append(simd_row)
    
    return features^, labels^


struct LogisticRegression:
    var w:row_type
    var b:Float32
    var lr: Float32
    var iter: Int

    fn __init__(out self, lr: Float32 = 0.0001, iter: Int = 10):
        self.lr = lr
        self.iter = iter
        self.w = row_type(0,0,0,0,0,0,0,0)
        self.b = Float32(0)

    fn _sigmoid(self, z: Float32)-> Float32:
        return 1/( 1 + math.exp(-z))

    fn __call__(self, x:row_type)-> Float32:
        z = (x * self.w).reduce_add() + self.b
        a = self._sigmoid(z)
        return a
    
    fn __call__(self, batch:List[row_type])-> List[Float32]:
        var preds:List[Float32] = List[Float32]()
        for x in batch:
            z = (x * self.w).reduce_add() + self.b
            a = self._sigmoid(z)
            preds.append(a)
        return preds^

    fn fit(mut self, X:List[row_type], Y: List[Float32]):
        var dw = row_type()
        var db = Float32(0)
        var m = len(X)
        for i in range(self.iter):
            for x,y in zip(X,Y):
                z = (x*self.w).reduce_add() + self.b
                a = self._sigmoid(z)
                err = -(y * math.log(a)) - (1-y)*(1-math.log(a))

                dw += err * x
                db += err
            
            self.w -= self.lr*(dw/m)
            self.b -= self.lr*(db/m)

            loss = self.loss_fn(X,Y)
            if (i+1)%10 == 0:
                try:
                    print(String("Epoch#{}; Loss:{}").format(i,loss))
                except: 
                    print("Unknow error")



    fn loss_fn(self, X:List[row_type], Y: List[Float32])->Float32:
        var m = len(X)
        var cost_sum = Float32(0)

        for x,y in zip(X,Y):
            z = (x*self.w).reduce_add() + self.b
            a = self._sigmoid(z)
            cost_sum += -(y * math.log(a)) - (1-y)*(1-math.log(a))

        cost_sum /= m
        return cost_sum


def main():
    data = load_data("./././data/heart_disease.csv")
    features = data[0].copy()
    labels = data[1].copy()

    var model: LogisticRegression = LogisticRegression(lr=0.000001, iter=100)
    model.fit(features, labels)

    print(model(features[:5]).__str__())