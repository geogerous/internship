import numpy as np
import struct
import os
# 分辨率
res = 128 
folder = "./Data"

for i in range(33):
    # 自动识别云或物体
    prefix = "cloud_" if i < 26 else "object_"
    idx = i if i < 26 else i - 26
    txt_path = f"{folder}/{prefix}{idx}.txt"
    bin_path = f"{folder}/{prefix}{idx}.bin"

    if os.path.exists(txt_path):
        print(f"Converting {txt_path}...")
        # 读取文本
        data = np.fromfile(txt_path, sep=" ", dtype=np.float32)
        
        # 严格匹配你的 C++ 代码结构：先写 resolution (int), 再写数据 (float)
        with open(bin_path, "wb") as f:
            f.write(struct.pack("i", res)) # 对应 fread(&resolution, ...)
            data.tofile(f)                 # 对应 fread(datas, ...)