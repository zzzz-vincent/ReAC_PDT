# Pedestrian Detection and Tracking
## PDT with RGB-D Camera
## PDT with LiDAR
### Model and Architecture
- Detection: adpated from Person-MinkUNet (https://github.com/VisualComputingInstitute/Person_MinkUNet)

	"The input to Person-MinkUNet is voxelized point cloud. In this work, we used voxel size (0.05m, 0.05m, 0.1m). A backbone network, implementation taken from [3], is used to extract features for each non-empty voxels. It is a submanifold sparse convolution network with ResNet20 architecture and U-Net connections. A fully connected layer is then used to regress 3D bounding boxes from the extracted features. These box proposals, after non-maximum suppression, are directly used as detections, with no refinement stage."[1]

- Tracking

	#TODO

### Dataset
- JRDB Dataset (https://jrdb.erc.monash.edu/)
- KITTI Semantic (http://www.semantic-kitti.org/)
### Reference
1. Jia, Dan, and Bastian Leibe. "Person-MinkUNet: 3D Person Detection with LiDAR Point Cloud." arXiv preprint arXiv:2107.06780. 2021.
2. Lang, Alex H., et al. "Pointpillars: Fast encoders for object detection from point clouds." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
3. Yin, Tianwei, Xingyi Zhou, and Philipp Krahenbuhl. "Center-based 3d object detection and tracking." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
4. Shi, Shaoshuai, et al. "PV-RCNN++: Point-voxel feature set abstraction with local vector representation for 3D object detection." arXiv preprint arXiv:2102.00463. 2021.
5. Shi, Shaoshuai, et al. "Pv-RCNN: Point-voxel feature set abstraction for 3d object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
6. Jia, Dan, and Bastian Leibe. "Person-MinkUNet: 3D Person Detection with LiDAR Point Cloud." arXiv preprint arXiv:2107.06780. 2021.
7. Zhou, Yin, and Oncel Tuzel. "VoxelNet: End-to-end learning for point cloud based 3d object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.


