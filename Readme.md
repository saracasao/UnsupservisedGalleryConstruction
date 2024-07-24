## Self-adaptive gallery for open-world person re-identification

### Installation

```
python3.8 -m venv gallery
cd gallery 
source bin/activate
git clone <repository git path>
cd <folder repo> 
pip install -r requirements.txt
```

### Quick guide 
The proposed method needs as input three types of data: images, features and skeletons.

By default, it is defined the unsupervised gallery construction with the _DukeMTMC-VideoReID_ dataset 
. Their data can be found in [here](
https://unizares-my.sharepoint.com/:u:/g/personal/scasao_unizar_es/EXoLTqEGh3lHigJR7KtWD0sBEww96BWEP2Elr9VEizwyxQ?e=Q27LbP).
You should extract the .zip file inside the ./data folder obtaining a directory structure as follows: 
```
./data
    |- DukeDataset
        |-DukeMTMC-VideoReID
            |-gallery
            |-query
            |-train
            |-LICENSE_DukeMTMC.txt
            |-LICENSE_DukeMTMC-VideoReID.txt
        |-DukeMTMC-VideoReID-Feat
            |-gallery
            |-query
            |-gallery_sklt
```
The first folder (DukeMTMC-VideoReID) contains the images and the second folder
(DukeMTMC-VideoReID-Feat) contains the appearance features (gallery, query) and the skeletons of the gallery (gallery_sklt)

The paths are defined in the scripts _main.py_ and _Settings.py_. Any issue finding the files you should check them.