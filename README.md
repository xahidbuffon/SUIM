*Repository for the paper [Semantic Segmentation of Underwater Imagery: Dataset and Benchmark](https://arxiv.org/pdf/2004.01241.pdf) (IROS 2020)* 
![det-data](/data/samples.jpg)

####  • [Paper (ArXiv)](https://arxiv.org/pdf/2004.01241.pdf)  • [Virtual Talk](https://youtu.be/LxWrhVeIkdg)  • [SUIM Dataset](http://irvlab.cs.umn.edu/resources/suim-dataset)  • [Experimental data](https://drive.google.com/drive/folders/1-ZGptUKC-yNFGxvOp207077_-Sf-VPOg?usp=sharing)  • [Checkpoints](https://drive.google.com/drive/folders/1aoluekvB_CzoaqGhLutwtJptIOBasl7i?usp=sharing) 

### SUIM Dataset
- For semantic segmentation of natural underwater images
- 1525 annotated images for training/validation and 110 samples for testing
- **BW**: Background/waterbody • **HD**: human divers • **PF**: Aquatic plants and sea-grass • **WR**: Wrecks/ruins
- **RO**: Robots/instruments   • **RI**: Reefs/invertebrates • **FV**: Fish and vertebrates • **SR**: Sea-floor/rocks


### SUIM-Net Model
- A fully-convolutional encoder-decoder network
- SUIM-Net (RSB): simple and light model; offers reasonable performance at a fast rate 
- SUIM-Net (VGG): provides better generalization performance 
- Detailed architecture is in [models](/models/); associated train/test scripts are also provided
- The [get_f1_iou.py](get_f1_iou.py) script is used for performance evaluation 


### Benchmark Evaluation
- Performance analysis for semantic segmentation and saliency prediction
- SOTA models in comparison: • [FCN](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) • [UNet](https://arxiv.org/pdf/1505.04597.pdf) • [SegNet](https://arxiv.org/pdf/1505.07293.pdf) • [PSPNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf) • [DeepLab-v3](https://arxiv.org/pdf/1706.05587.pdf) 
- Metrics: • region similarity (F score) and • contour accuracy (mIOU)
- Download experimental data [from here](https://drive.google.com/drive/folders/1-ZGptUKC-yNFGxvOp207077_-Sf-VPOg?usp=sharing) and checkpoints data [from here](https://drive.google.com/drive/folders/1aoluekvB_CzoaqGhLutwtJptIOBasl7i?usp=sharing)

![det-data](/data/quan.png)
![det-data](/data/qual.png)


#### Bibliography Entry
	
	@inproceedings{islam2020suim,
	  title={{Semantic Segmentation of Underwater Imagery: Dataset and Benchmark}},
	  author={Islam, Md Jahidul and Edge, Chelsey and Xiao, Yuyang and Luo, Peigen and Mehtaz, 
                  Muntaqim and Morse, Christopher and Enan, Sadman Sakib and Sattar, Junaed},
	  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
	  year={2020},
	  organization={IEEE/RSJ}
	}
	

#### Acknowledgements
- https://github.com/qubvel/segmentation_models
- https://github.com/divamgupta/image-segmentation-keras
- https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap
- https://github.com/zhixuhao/unet
- https://github.com/aurora95/Keras-FCN
- https://github.com/MLearing/Keras-Deeplab-v3-plus/
- https://github.com/wenguanwang/ASNet

