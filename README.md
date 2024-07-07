paper名称是《MOWA: Multiple-in-One Image Warping Model》, ariv链接 https://arxiv.org/pdf/2404.10716
我第一次读这篇paper时，就被文章将开头的图片吸引了，图片是这样的。
![teaser](https://github.com/hpc203/MOWA-onnxrun/assets/28389623/5b5a8186-df0e-4ffe-bba2-b06eeccde5df)
可以看到，它一个模型就解决多个图像扭曲任务的，并且在摘要里的最后说：“大量实验表明，我们的 MOWA 在六个多合一图像扭曲任务上进行训练，在大多数任务上都优于最先进的特定任务模型。此外，MOWA 还表现出泛化到未见过场景的潜力，跨域和零样本评估证明了这一点。”


这么好的文章，我立刻就想着导出onnx模型文件，编写推理部署程序的，经过3天的编写和调试，最后发布了这套代码。
起初想用opencv-dnn做推理引擎的，可是opencv-dnn加载onnx文件报错了，看日志信息推测是在encoder模块里的LeWinTransformerBlock模块里的masked_fill在捣鬼
，因此我使用onnxruntime做推理殷勤。

在编写c++程序的过程中发现了一个有趣的现象，那就是在GridSamplerBilinear函数里，数组以vector的形式访问和赋值，耗时140秒，
可是以指向数组的指针形式访问和赋值，耗时只有0.003秒，相差了46666倍的速度。


onnx文件在百度云盘，链接：https://pan.baidu.com/s/17b_kNE9azY3gE3gnqXTXLw 
提取码：68i1


测试图片数量比较多，打包上传到百度云盘了，链接：https://pan.baidu.com/s/1Fq871r6TTsxcpqthgME9cg 
提取码：jxk6
