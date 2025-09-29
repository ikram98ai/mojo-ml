import random
import math
from python import Python


fn train(xbatch:List[Float32], ybatch:List[Float32], mut params: List[Float32], lr: Float32= 0.000001):
    dw:Float32 = 0.0
    db:Float32 = 0.0
    for x, y in zip(xbatch, ybatch):
        z = (x*params[0] + params[1]) - y
        dw += z*x
        db += z

    m = len(xbatch)
    dw = dw / m
    db = db / m

    params[0] = params[0] - lr * dw
    params[1] = params[1] - lr * db


fn cost_fn(xbatch:List[Float32], ybatch:List[Float32], params:List[Float32])-> Float32:
    cost_sum:Float32 = 0.0
    for x, y in zip(xbatch, ybatch):
        cost_sum += ((x*params[0] + params[1]) - y)**2
    m = len(xbatch)
    loss = cost_sum / (2 * m)
    return loss


def read_data(path:String) ->  Tuple[List[Float32], List[Float32]]:
    pl = Python.import_module("polars")

    df = pl.read_csv(path,schema_overrides={"BuildingArea": pl.Float32})
    df = df.drop_nulls(subset=["BuildingArea", "Price"])

    features:List[Float32] = [Float32(x) for x in df['BuildingArea']]
    labels:List[Float32] = [Float32(y) for y in df['Price']]

    return features^, labels^

def main():
    var features: List[Float32]
    var labels: List[Float32]

    data = read_data("././data/property_sales.csv")
    features = data[0].copy()
    labels = data[1].copy()


    random.seed()
    params = [Float32(1.0), Float32(1.0)]

    print("Length of dataset: ", len(features))
    # split = 150
    for i in range(10):
        train(features, labels, params)
        loss = cost_fn(features, labels, params)
        print("Loss#",i,": ",loss, " w: ", params[0], " b: ", params[1])