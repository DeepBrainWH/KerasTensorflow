import base64
import json
from os import listdir
from os.path import isfile, join

dir = "/home/wangheng/Downloads/资料下载/my_credit_card/"

l2 = [f for f in listdir(dir) if isfile(join(dir, f)) and f.endswith("json")]

# print(l2)

with open(dir+l2[0], "r") as f:
    print(l2[0])
    my_dict = json.load(f)
for key in my_dict:
    print(key)
data_ = my_dict['imageData']
fh = open(dir + "base64.png", "wb")
decode = base64.b64decode(data_)
fh.write(decode)
fh.close()

