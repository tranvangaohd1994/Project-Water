import glob
import os
listimgs = glob.glob("/home/vti/Downloads/captcha 2/train/*.png")
file1 = open("gt_train.txt","w") 
for path in listimgs:
    name = os.path.splitext(os.path.basename(path))[0]
    label = name
    L = [f"train/{name}.png\t{label}\n"]  
    file1.writelines(L) 
file1.close()