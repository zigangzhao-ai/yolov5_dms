'''
code by zzg 2020-05-13
'''

##xml_to_json

import glob
import xmltodict
import json

path = "/workspace/zigangzhao/yolov5_new1013/tool/dms1013/xml_modify/"
path2 = "/workspace/zigangzhao/yolov5_new1013/tool/dms1013/json/"

xml_dir = glob.glob(path + '*.xml')
print(xml_dir)

def pythonXmlToJson(path):
  
    xml_dir = glob.glob(path + '*.xml')
    # print(len(xml_dir))
    for x in xml_dir:
        with open(x) as fd:
            convertedDict = xmltodict.parse(fd.read())
            jsonStr = json.dumps(convertedDict, indent=1)
            print("jsonStr=",jsonStr)
            print(x.split('.')[0])
            json_file = x.split('.')[0].split('/')[-1] +'.json'
            with open(path2 + '/' + json_file, 'w') as json_file:
                json_file.write(jsonStr)
    print("xml_json finished!")
    print(len(xml_dir))
pythonXmlToJson(path)