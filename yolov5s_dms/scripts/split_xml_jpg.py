'''
code by zzg -2020-10-07
'''
#复制或者移动一个文件夹下的所有图片或者其他指定文件到另一个文件夹
import os
import shutil
path = 'dms_1013'
new_path = 'image/'
new_path1 = 'xml/'
count = 0
for root, dirs, files in os.walk(path):
    for i in range(len(files)):
        if (files[i][-3:] == 'jpg' or files[i][-3:] == 'JPG') or files[i][-4:]=='jpeg':
        #if (files[i][-3:] == 'xml'):
            count += 1
            file_path = root + '/' + files[i]
            new_file_path = new_path + '/' + files[i]
            shutil.copy(file_path, new_file_path)
            #shutil.move(file_path, new_file_path))

print(count)
print("move finished!!")
