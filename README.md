# Connecting Pixels to Privacy and Utility: Automatic Redaction of Private Information in Images

**Tribhuvanesh Orekondy, Mario Fritz, Bernt Schiele**    
Max Planck Institute for Informatics

For details, refer to our paper: 
[Connecting Pixels to Privacy and Utility: Automatic Redaction of Private Information in Images](http://openaccess.thecvf.com/content_cvpr_2018/papers/Orekondy_Connecting_Pixels_to_CVPR_2018_paper.pdf), CVPR 2018

## Abstract
Images convey a broad spectrum of personal information. If such images are shared on social media platforms, this personal information is leaked which conflicts with the privacy of depicted persons. Therefore, we aim for automated approaches to redact such private information and thereby protect privacy of the individual. By conducting a user study we find that obfuscating the image regions related to the private information leads to privacy while retaining utility of the images. Moreover, by varying the size of the regions different privacy-utility trade-offs can be achieved. Our findings argue for a redaction by segmentation paradigm. Hence, we propose the first sizable dataset of private images "in the wild" annotated with pixel and instance level labels across a broad range of privacy classes. We present the first model for automatic redaction of diverse private information. It is effective at achieving various privacy-utility trade-offs within 83% of the performance of manual redaction.

## Dataset
Coming soon...

## Training - Segmentation
Refer to the scripts within `models/` directory.

## Evaluation - Segmentation

### Output format
Prepare the predictions as a single JSON file represeting output over the entire test set.
This JSON file should represent a list of predicted regions (encoded as COCO RLE):
```
[
  {
    "image_id": "2017_19919173",
    "segmentation": {
      "counts": "0Pb^;",
      "size": [
        510,
        736
      ]
    },
    "score": 0.9996910095214844,
    "attr_id": "a105_face_all"
  },
  ...
]
```

### Running evaluation
Evaluation is run in two steps.
First, compute the precision/recall of predictions over multiple thresholds (50 in this case):
```
python privacy_filters/tools/evaltools/evaluate_curves.py path/to/gt/test2017.json path/to/predicted.json -w -n 50
```
This will prepare a `predicted-curves.json` within the same predicted directory.
Use this in the next step of calculating Average Precision (area under the precision-recall curve):
```
python privacy_filters/tools/evaltools/calc_ap.py path/to/predicted-curves.json
```

## Contact
For any problems, please contact [Tribhuvanesh Orekondy](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/tribhuvanesh-orekondy/) (orekondy at mpi-inf dot mpg dot de)

## Citation
```
@InProceedings{Orekondy_2018_CVPR,
author = {Orekondy, Tribhuvanesh and Fritz, Mario and Schiele, Bernt},
title = {Connecting Pixels to Privacy and Utility: Automatic Redaction of Private Information in Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
