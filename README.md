#### Pointers: 
- [Paper (arxiv preprint)](soon)  • [Dataset](http://irvlab.cs.umn.edu/resources/suim-dataset)  • [Experimental data](https://drive.google.com/drive/folders/1-ZGptUKC-yNFGxvOp207077_-Sf-VPOg?usp=sharing)

#### SUIM Dataset
- For semantic segmentation of natural underwater images
- 1525 annotated images for training/validation and 110 samples for testing
- **BW**: Background/waterbody • **HD**: human divers • **PF**: Aquatic plants and sea-grass • **WR**: Wrecks/ruins
- **RO**: Robots/instruments   • **RI**: Reefs/invertebrates • **FV**: Fish and vertebrates • **SR**: Sea-floor/rocks
![det-data](/data/samples.jpg)


#### SUIM-Net Model
- A fully-convolutional encoder-decoder network: embodies residual learning and mirrored skip connection
- Offers competitive semantic segmentation performance at a fast rate (**28.65 FPS** on a 1080 GPU) 
- See [model.py](model.py) for details; associated train/test scripts are also provided
- Checkout the [get_f1_iou.py](get_f1_iou.py) script for performance evaluation 


#### Benchmark Evaluation
- Performance analysis for semantic segmentation and saliency prediction
- SOTA models in comparison: • FCN • UNet • SegNet • PSPNet • DeepLab-v3 
- Metrics: • region similarity (F score) and • contour accuracy (mIOU)
- Further analysis and implementation details are provided in the paper

![det-data](/data/quan.png)
![det-data](/data/qual.png)


### Acknowledgements
- https://github.com/qubvel/segmentation_models
- https://github.com/divamgupta/image-segmentation-keras
- https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap
- https://github.com/zhixuhao/unet
- https://github.com/aurora95/Keras-FCN
- https://github.com/MLearing/Keras-Deeplab-v3-plus/
- https://github.com/wenguanwang/ASNet

