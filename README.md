# Code Repository for SeqNet Paper

## Requierments

> Python = 3.10,
> Tensorflow = 2.10,
> Scikit-Learn,
> Cuda-Toolkit = 11.2,
> Cudnn = 8.6.0

## Dataset

[Download dataset](https://drive.google.com/file/d/178HDL8RhwbH2AbdT0PA_efMnmnJpJiOC/)

## H5 files

To use our trained models, download and extract model h5 files from this [link](https://drive.google.com/file/d/1bDQk5-POFjlDYRG1TMRclFOXpme_wfY1/)

### For meta-training, run this codeshell

```shell
python train.py --backbone <name of backbone> --test <test data>
--epochs <number of epochs> --batch <batch size> --lr <learning rate>
--dim <input dimentsion>
```



> \<test data>
> 
> gtsrb2tt100k: $\text{GTSRB}\to \text{TT100K}$ (default)
> 
> gtsrb: $\text{GTSRB}\to \text{GTSRB}$
> 
> belga2flick: $\text{Belga}\to \text{Flickr32}$
> 
> belga2toplogo: $\text{Belga}\to \text{Toplogo10}$
> 
> gtsrb2flick: $\text{GTSRB}\to \text{Flickr32}$
> 
> gtsrb2toplogo: $\text{GTSRB}\to \text{Toplogo10}$

> \<name of backbone>
> 
> densenet (default),
> densenetmini,
> resnet,
> mobilenet

### Nearest neighbor evaluation

> python inference.py --backbone \<name of backbone> --data \<test data> --device \<device name> --dist \<distance metric>

For instance, to run $\text{GTSRB}\to \text{TT100K}$ evaluation on the cpu with cosine distance metric, run the following script:

> python inference.py --data gtsrb2tt100k --backbone densenet --device cpu --dist cosine

### Traditional GTSRB benchmark run this code

> python gtsrb-benchmark.py --mode \<mode name> --epochs \<number of epochs> 
> 
> \<mode name>: train: training, test: evaluation

## Video Showcase for Real-World Scenario

https://github.com/narimanabdi/seqnet/assets/158992158/c458c6ad-6cf5-496d-b34f-16645a704777


