# Task2 数据读取与扩增

## Pillow
- Resize()与Thumbnail()方法对比

    - resize()方法可以缩小也可以放大，而thumbnail()方法只能缩小

    - resize()方法不会改变对象的大小，只会返回一个新的Image对象，而thumbnail()方法会直接改变对象的大小，返回值为none

    - resize()方法中的size参数直接规定了修改后的大小，而thumbnail()方法按比例缩小，size参数只规定修改后size的最大值。

<div align='center'>
    <img src='img/thumbnail.jpg' />
</div>

---

## OpenCV

- COLOR__BGR2RGB()：将BGR图片转化为RGB图片，RGB中R在高位，G在中间，B在低位。BGR正好相反。

- COLOR_BGR2GRAY()：将BGR转化为灰度图片

- cv2.Canny(src, thresh1, thresh2)：提取图像中物体的边缘

    - src表示输入的图片

    - thresh1表示最小阈值

    - thresh2表示最大阈值

较大的阈值thresh2用于检测图像中明显的边缘，但一般情况下检测的效果不会那么完美，边缘检测出来是断断续续的。所以这时候用较小的第一个阈值用于将这些间断的边缘连接起来。

<div align='center'>
    <img src='img/test.jpg' />
</div>

---

## Pytorch中的数据扩增

以torchvision为例，常见的数据扩增方法包括：

- transforms.CenterCrop：对图片中心进行裁剪      
- transforms.ColorJitter：对图像颜色的对比度、饱和度和零度进行变换      
- transforms.FiveCrop：对图像四个角和中心进行裁剪得到五分图像     
- transforms.Grayscale：对图像进行灰度变换    
- transforms.Pad：使用固定值进行像素填充     
- transforms.RandomAffine：随机仿射变换    
- transforms.RandomCrop：随机区域裁剪     
- transforms.RandomHorizontalFlip：随机水平翻转     
- transforms.RandomRotation：随机旋转     
- transforms.RandomVerticalFlip：随机垂直翻转   

    