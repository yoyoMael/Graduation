#! python2
# -*- coding:utf-8 -*-
from PIL import Image
import struct
def read_image(filename):
  f = open(filename, 'rb')

  index = 0
  buf = f.read()

  f.close()

  magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
  # 读取魔数等
  index += struct.calcsize('>IIII')
  # 增加索引值
  for i in xrange(images):
    image = Image.new('L',(columns, rows))
    # ‘L’创建灰度图，第二个参数为图片大小（列，行）
    for x in xrange(rows):
      for y in xrange(columns):
        image.putpixel((x,y), int(struct.unpack_from('>B', buf, index)[0]))
        # 读取灰度值，用putpixel写入image对象
        index += struct.calcsize('>B')

    print 'save' + str(i) + 'image'
    image.save(str(i) + '_c.png')
    # 将image保存

def read_label(filename, saveFilename):
  f = open(filename, 'rb')
  index = 0
  buf = f.read()

  f.close()

  magic, labels = struct.unpack_from('>II', buf, index)
  index += struct.calcsize('>II')

  labelArr = [0] * labels

  for x in xrange(labels):
    labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
    index += struct.calcsize('>B')

    save = open(saveFilename, 'w')

    save.write(','.join(map(lambda x: str(x), labelArr)))

    save.close()
    print 'save labels success'


if __name__ == '__main__':
  read_image('../t10k-images-idx3-ubyte')
  # read_label('../t10k-labels-idx1-ubyte', 'label.txt')


