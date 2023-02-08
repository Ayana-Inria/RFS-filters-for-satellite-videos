# RFS-filters for Satellite Videos
RFS filters (GLMB and GM-PHD)

This code was used to produce the results shown in:

> C. Aguilar, M. Ortner and J. Zerubia, "ENHANCED GM-PHD FILTER FOR REAL TIME SATELLITE MULTI-TARGET TRACKING
," 2023"

If you use this code, we strongly suggest you cite:

    @inproceedings{aguilar2023,
        author = {Aguilar, Camilo and Ortner, Mathias and Zerubia, Josiane},
        title = {ENHANCED GM-PHD FILTER FOR REAL TIME SATELLITE MULTI-TARGET TRACKING},
        booktitle = {2023},
        pages={},
        doi={},
        Year = {2023}
    }


|Tracked Objects | 
|:--:| 
| <img src="docs/afrl2.gif">|

### RFS Filters

### Contents
1. [Installation](#Installation)
2. [Usage](#usage)
3. [Dataset Registration](#dataset)


### Installation

1. Clone the repository
  ```Shell
  git clone --recursive https://github.com/Ayana-Inria/RFS-filters-for-satellite-videos
  ```

2. To install required dependencies run:
```Shell
$ pip install requirements.txt
```
Required libraries
```Shell
- numpy=1.22
- pytorch=1.12
- opencv=4.5
- pillow=9.0
- matplotlib=3.5
- motmetrics=1.2
```

### Usage

To run the demo
```Bash
python demo.py
```

Demo outputs are saved under:

```
[root directory]/OUTPUTS
```


#### Using RF-Filters in your own project


### Dataset
#### Video Regisration
1. Download the WPAFB 2009 dataset from the [AFRL's Sensor Data Management System (SDMS) website](https://www.sdms.afrl.af.mil/index.php?collection=wpafb2009)
2. Convert the .ntf files to .png (we used Matlab's _nitfread_ function)
3. Use our [video stabilization repository](https://github.com/Ayana-Inria/satellite-video-stabilization) to stabilize the sequence and format the labeling


#### Data Structure
To replicate the results shown in the paper, the DATASET needs to be formatted in the following way:
```
[root for demo.py]
└──dataset
      └── WPAFB_2009/
            └── AOI_02/
                ├── INPUT_DATA/
                |   ├── img01.png
                |   ├── img02.png
                |   └── ...
                ├── GT/
                |   ├── stabilized_oject_states.csv
                |   └── labels/
                |         ├── labels_as_points_01.png
                |         ├── labels_as_points_02.png
                |         └── ...
                └── FILTER_OUTPUT/
                    ├── birth_field/
                    |     ├── birth_field_01.png
                    |     ├── birth_field_02.png
                    |     └── ...
                    ├── objects/
                    |     ├── tracked_objects_01.png
                    |     ├── tracked_objects_02.png
                    |     └── ...
                    ├── labels/
                    |     ├── labels_as_points_01.png
                    |     ├── labels_as_points_02.png
                    |     └── ...
                    └── object_states.csv
```

### Aknowledgment
    Thanks to BPI France (LiChiE contract) for funding this research work, and to the OPAL infrastructure from Université Côte d'Azur for providing computational resources and support.

### License
    GLMB with adaptive birth is released under the GNUv3 License (refer to the LICENSE file for details).