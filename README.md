#Code Repository for SENet Paper

##Requierments

 
>Python = 3.10,
Tensorflow = 2.10,
Scikit-Learn,
Cuda-Toolkit = 11.2,
Cudnn = 8.2
>

to meta-train run this code
> python train.py --backbone \<name of backbone> --test \<test data>
--epochs \<number of epochs> --batch \<batch size> --lr \<learning rate>
--dim \<input dimentsion>

>\<test data>
>
>gtsrb2tt100k: $GTSRB\to TT100K$ (default)
>
>gtsrb: $GTSRB\to GTSRB$
>
>belga2flick: $Belga\to Flickr32$
>
>belga2toplogo: $Belga\to Toplogo10$
>
>gtsrb2flick: $GTSRB\to Flickr32$
>
>gtsrb2toplogo: $GTSRB\to Toplogo10$

>\<name of backbone>
>
>densenet (default),
>densenetmini,
>resnet,
>mobilenet


to run nearest neighbor evaluation
> python inference.py --backbone \<name of backbone> --data \<test data> --device \<device name> --dist \<distance metric>


to traditional GTSRB benchmark run this code
> python gtsrb-benchmark.py --mode \<mode name> --epochs \<number of epochs> 
>
> \<mode name>: train: training, test: evaluation

