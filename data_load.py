import gzip 
import numpy as np 

with gzip.open('./data/train-images-idx3-ubyte.gz') as f:
    magic_no = f.read(4)
    magic_no = int.from_bytes(magic_no, "big")
    image_no = int.from_bytes(f.read(4), "big")
    row_no = int.from_bytes(f.read(4), "big")
    col_no = int.from_bytes(f.read(4), "big")
    print(magic_no, " ", image_no, " ", row_no,  " " ,col_no)
    dataset = []
    for i in range(60000):
        b =[i for i in  f.read(28*28)]
        dataset.append(b)

image_data = np.array(dataset, dtype=np.int16)
print(image_data.shape)

with open("./data/image_data.npy", "wb") as f:
    np.save(f, image_data)