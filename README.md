#Code Repository for SENet Paper

##Requierments

 
>Python = 3.9
Tensorflow = 2.10
Scikit-Learn
Cuda-Toolkit = 11.2
Cudnn = 8.2
>

to meta-train run this code
> python run.py --mode metatrain --epochs \<number of epochs> --test \<name of experiment>

>\<name of experiment>
>gtsrb2tt100k: $GTSRB\to TT100K$ (default)
gtsrb: $GTSRB\to GTSRB$
belga2flick: $Belga\to Flickr32$
belga2toplogo: $Belga\to Toplogo10$
gtsrb2flick: $GTSRB\to Flickr32$
gtsrb2toplogo: $GTSRB\to Toplogo10$

to run nearest neighbor evaluation
> python nn-test.py --test \<name of experiment> --mode \<mode name> 


>\<mode name>
>base: DenseNet121 TFE trained on ImageNet (default)
resnet: ResNet50 TFE trained on ImageNet (only $GTSRB\to TT100K$)
mobilenet: MobileNet TFE trained on ImageNet (only $GTSRB\to TT100K$)
mini: DenseNet121 TFE trained on miniImageNet (only $GTSRB\to TT100K$)
random: DenseNet121 TFE random initialization (only $GTSRB\to TT100K$)

to traditional GTSRB benchmark run this code
> python gtsrb_benchmark.py --mode \<mode name> --epochs \<number of epochs> 
> \<mode name>: train: training, test: evaluation

