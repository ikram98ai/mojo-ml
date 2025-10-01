from python import Python
import math 

alias simd_width = 8
alias row_type = SIMD[DType.float32, simd_width]

def load_data(file_path:String)-> (List[row_type], List[Float32]):
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
    var wd: Float32
    var iter: Int

    fn __init__(out self, lr: Float32 = 0.01, wd:Float32= 0.1, iter: Int = 10):
        self.lr = lr
        self.wd = wd
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
                err = -(y * math.log(a)) -(1-y)*math.log(1-a)

                # for j in range(simd_width):
                #     dw[j] = dw[j] + err * x[j]
                dw += err * x
                db += err
            
            # for j in range(simd_width):
            #     self.w[j] = self.w[j] - (self.lr * dw[j] / m) + (self.wd/m ) * self.w[j]
            self.w -= self.lr*(dw/m) + (self.wd/m)*self.w
            self.b -= self.lr*(db/m)

            loss = self.loss_fn(X,Y)
            if (i+1)% (self.iter*0.1) == 0:
                try:
                    print(String("Epoch#{}; Loss:{}").format(i+1,loss))
                except: 
                    print("Unknow error")
                    
    fn loss_fn(self, X:List[row_type], Y: List[Float32])->Float32:
        var m = len(X)
        var cost_sum = Float32(0)

        for x,y in zip(X,Y):
            z = (x*self.w).reduce_add() + self.b
            a = self._sigmoid(z)
            cost_sum += -(y * math.log(a)) - (1-y) * math.log(1-a)

        reg_cost = (self.w**2).reduce_add()

        cost = (cost_sum/m ) + (self.wd/(2*m)*reg_cost)
        return cost


def main():
    data = load_data("./././data/heart_disease.csv")
    features = data[0].copy()
    labels = data[1].copy()

    var model = LogisticRegression(lr=0.0001, wd=0.009,iter=20)
    model.fit(features, labels)

    print(model(features[:5]).__str__())