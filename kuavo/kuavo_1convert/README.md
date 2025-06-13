# data and train
## data convert from rosbag to zarr
### 1. init content dataset strucutre:
```
PortableSSD(dataset)
├── dataset
│   └── Task2-RearangeToy
│       └── kuavo-rosbag
│           ├── task_pcd_test_2024-12-27-21-55-02.bag
│           └── task_pcd_test_2024-12-27-22-04-02.bag
    
```

### 2. run command:  
**before run command, you need modify/add a new yaml file like`Task5-GrabB.yaml` according to your rosbag topic and message type.**
```bash
bash data-train-deploy/src/kuavo-1convert/rosbag2zarr.sh -b /media/lab/PortableSSD/DATASET/Task5-GrabB/kuavo-rosbag -c /home/lab/hanxiao/dataset/kuavodatalab/data-train-deploy/src/kuavo-1convert/Task5-GrabB.yaml -n 102 -a
```

### 3. after convert content strucutre:
```
PortableSSD(dataset)
├── dataset
│   └── Task2-RearangeToy
│       ├── kuavo-rosbag
│       ├── kuavo-zarr
│       ├── plt-check
│       ├── raw-video
│       └── sample-video

```



-----------------------------



## train with zarr data
1. modify/add `config` in `/home/lab/hanxiao/dataset/kuavodatalab/data-train-deploy/src/util/diffusion_policy/diffusion_policy/config` according to your data
    - task config: KuavoGrabB_task.yaml
        - dataset_path(you .zarr file path)
        - obs and action shape
        - dataset loader Class _target_

    - training config: train_diffusion_unet_real_image_workspace_KuavoToy_task.yaml
        - task(according to your task config file name)
        - some other params
        
2. modify/add `dataset` in `/home/lab/hanxiao/dataset/kuavodatalab/data-train-deploy/src/util/diffusion_policy/diffusion_policy/config` according to your data
    - train dataset loader: /home/camille/IL/diffusion/diffusion_policy/dataset/pusht_image_dataset_KuavoToy_task.py

3. run command:
```bash
python /home/camille/IL/diffusion/train.py --config-name train_diffusion_unet_real_image_workspace_KuavoToy_task
```

## deploy(TODO)