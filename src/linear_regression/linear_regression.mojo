from python import Python 

alias width = 8

def load_data(file_path:String)-> Tuple[ List[SIMD[DType.float32, width]], List[Float32] ]:
    pl = Python.import_module("polars")
    df = pl.read_csv(file_path)
    df = df.cast(pl.Float32)
    features_df = df.select(pl.all().exclude("target", "chol"))

    var features = List[SIMD[DType.float32, width]]()
    for row in features_df.iter_rows():
        var row_items = SIMD[DType.float32, width]()
        for i in range(width):
            row_items[i] = Float32(row[i])
        features.append(row_items)

    labels = [Float32(y) for y in df['chol']]

    return features^, labels^

fn train_linear(features: List[SIMD[DType.float32, width]], 
                labels:List[Float32], 
                mut w:SIMD[DType.float32, width], 
                mut b:Float32, lr: Float32 = 0.0003)-> 
                Tuple[SIMD[DType.float32, width], Float32, Float32]:

 
    var m = len(features)

    var dw = SIMD[DType.float32, width]()
    var db = Float32(0)
    var cost_sum = Float32(0.0)

    for x,y in zip(features,labels):
        err = ((x*w).reduce_add() + b) - y 
        cost_sum += err**2

        dw += err * x
        db += err

    loss = cost_sum/(2*m)
    dw = dw/m
    db = db/m

    w -= lr * dw
    b -= lr * db

    return w, b, loss


def main():
    var features: List[SIMD[DType.float32, width]] 
    var labels: List[Float32]

    data = load_data("././data/heart_disease.csv")   
    var w = SIMD[DType.float32, width](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var b = Float32(0.0)
    features = data[0].copy()
    labels = data[1].copy()
    for i in range(10):
        w, b, loss = train_linear(features,labels, w, b)
        print(String("Epoch:{}; Loss:{}; W:{}; b:{}").format(i,loss,w,b))
