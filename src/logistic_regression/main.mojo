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

fn sigmoid(z: Float32)-> Float32:
    return 1/( 1 + math.exp(-z))

fn train_logistic(features:List[row_type], labels: List[Float32], mut w: row_type, mut b: Float32, lr: Float32 = 0.0001):
    var dw = row_type()
    var db = Float32(0)
    var m = len(features)

    for x,y in zip(features,labels):
        z = (x*w).reduce_add() + b
        a = sigmoid(z)
        err = -(y * math.log(a)) - (1-y)*(1-math.log(a))

        dw += err * x
        db += err
    
    w -= lr*(dw/m)
    b -= lr*(db/m)

fn loss_fn(features:List[row_type], labels: List[Float32], mut w: row_type, mut b: Float32)->Float32:
    var m = len(features)
    var cost_sum = Float32(0)

    for x,y in zip(features,labels):
        z = (x*w).reduce_add() + b
        a = sigmoid(z)
        cost_sum += -(y * math.log(a)) - (1-y)*(1-math.log(a))

    cost_sum /= m
    return cost_sum


def main():
    data = load_data("./././data/heart_disease.csv")
    features = data[0].copy()
    lables = data[1].copy()

    var w = row_type(0,0,0,0,0,0,0,0)
    var b = Float32(0)

    for i in range(10):
        train_logistic(features, lables, w, b)
        loss = loss_fn(features, lables, w, b)
        print(String("Epoch#{}; Loss:{}; w:{}; b:{}").format(i,loss,w,b))
        
