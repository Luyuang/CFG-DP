### Model/Train
1. camhl + low40: 较小的low_state波动导致模型预测随机；camhl + low13/low9/low8, 模型受low_state波动影响小；
2. batch_size小一点，在抓取动作差异大时，能学到对应的动作；
3. batch_size = 8 时， 15mins/10000steps

### dataset
1. resize to one cam shap
2. eps长度要一致，保证速度相同；避免重复抓取；
3. 开/合，每次手的状态一样；

### code 
## LeRobotEnv:
1. 30hz在时间戳对齐时，两个相机时间戳差一两帧，导致按照最新帧对齐时，有一个相机取两个timestep是同一帧

### TODO
1. 把切片参数加到模型的config中

