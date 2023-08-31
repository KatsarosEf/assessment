 # Smarter DiagnX Assessment 

The following project leverages patient knee MRI slices to classify knee malfunctions.

## Installation

```
git clone https://gitlab.com/smarter-diagnx/assessment.git
cd assessment/
```
Make a virtual environment with ```virtualenv```
```
virtualenv -p python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset
Download the dataset accompanying the MRNet publication and move it to `./data`. The data structure should be as follows:
```Shell
  data/
      train/
          axial/
          sagittal/
          coronal/
      valid/
          axial/
          sagittal/
          coronal/
      train-abnormal.csv
      train-acl.csv
      train-meniscus.csv
      valid-abnormal.csv
      valid-acl.csv
      valid-meniscus.csv
```  

## Train
The dataset can be utilized to classify MRI into abnormal, ACL or meniscus, that is, three different classification tasks.
To do so, one could use one input plane i.e. axial, or fuse information from more planes, i.e. both axial and sagittal. 
To train the respective models, please use the following commands:

```python train.py -t abnormal -p axial --out axial```

or

``` python train.py -t abnormal -p axial -f sagittal --out axial_sagittal```

respectively. More generally, `-p` specifies the input plane (required) and 
`-f` (fusion) specifies the second input plane to be late-fused and enhance the
learnt MRI knee feature representations. In each case, the model weights are stored in the directories
specified by ```--out```.

The results using the trained models on the validation set are the following:

| Planes           | AUC    | Accuracy | F1    | @epoch |
|------------------|--------|----------|-------| ------ |
| Axial            | 0.9229 | 0.858    | 0.916 | 36     |
| Axial & Sagittal | 0.9373 | 0.900    | 0.940 | 28     |

The training curves can be monitored at [wandb](https://wandb.ai/ekatsaros/mri-cv/reports/Knee-MRI-training-logs--VmlldzoxNzA4MTcy).
## Evaluate
To reproduce the results, please download the [single](https://drive.google.com/file/d/1_eJpqxFdWOq0pIiuZSoenDP5BpL3zCNS/view?usp=sharing) and [fused](https://drive.google.com/file/d/1ZqixrNImNZeD33fnmbGMiJJlt1FzQDU6/view?usp=sharing) models. 
Store the single model at `./models/axial`
and the fused one at `./models/axial&sagittal`. Thereafter, to validate results for "Axial" please use:

```python evaluate.py -p axial```

Similarly, to reproduce results for "Axial & Sagittal" please use

```python evaluate.py -p axial -f sagittal```

## Unit tests

Last, to run the unit tests, please run:

```python -m unittest discover -s tests -p "test_*"```

Please note the tests are written as a proof of concept. They assess the dataloader shapes and the model output shapes.
Last, they check whether the probabilities from the model outputs are in between 0-1.
Clearly, the code can be easily extended for more elaborate testing.

## Run the test script within a docker container
To build the image from docker please run:
```docker build -t test .```
where "test" is the image name.
Once the image is built, execute the following to make a container out of the created image and run the test script inside:
```docker run -i -t test python evaluate.py -p axial```
or 
```docker run -i -t test python evaluate.py -p axial -f sagittal```.
Please make sure that the trained models reside already within the `./models/axial` and `./models/axial&sagittals`
directories BEFORE building the docker image.