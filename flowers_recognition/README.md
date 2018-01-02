## REF
[agoila/transfer-learning](https://github.com/agoila/transfer-learning)


## usage
https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy
http://download.tensorflow.org/example_images/flower_photos.tgz

#### step0:
Download the vgg16 model file and flower_photos.tgz, then untar it.

```shell
cd flower_recognition/
python download.py
```

#### step1:
Process the photo files to tfrecords

```shell
python image_processing.py
```
#### step3:
Train the model

~~~shell
python training.py
~~~
#### step3:
Predict the image

~~~shell
python prediction.py
~~~