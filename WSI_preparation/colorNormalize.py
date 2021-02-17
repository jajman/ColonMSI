import staintools, os
from PIL import Image

def getFileName(fileName):
    lenFIleName = len(fileName)
    nameEnd = 0
    nameStart = 0
    isDot=False
    for i in reversed(range(lenFIleName)):
        if fileName[i] == '.':
            if isDot:
                continue
            else:
                nameEnd = i
                isDot=True
        if fileName[i] == '/':
            nameStart = i + 1
            break
    return fileName[nameStart:nameEnd]


target = staintools.read_image("tissue.png")


normalizer = staintools.ReinhardColorNormalizer()
normalizer.fit(target)

#staintools.plot_image(transformed)


srcFolder='/home/jajman/validation-TCGAColonMSI/normal/'
dstFolder ='/home/jajman/validation-TCGAColonMSI_CN/normal/'
files = os.listdir(srcFolder)

for i in files:
    to_transform = staintools.read_image(srcFolder+i)
    transformed = normalizer.transform(to_transform)
    img = Image.fromarray(transformed)
    img.save(dstFolder+i)
