import sys
import os

folderName=sys.argv[1]
imageList=[]
if(os.path.isdir(folderName)):
    imageList= [os.path.join(folderName, f) for f in os.listdir(folderName)]
else:
    imageList.append(folderName)


print(imageList)