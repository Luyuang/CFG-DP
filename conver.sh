#############使用前将数据集文件放到kuavo_il下且重命名为testbag即可
set -e
IMAGE_NAME="dataset_generator"
CONTAINER_NAME="temp_container"
HOST_OUTPUT_DIR="./conver"

echo "[1/3] 构建 Docker 镜像..."
# docker build -t $IMAGE_NAME .
echo "[2/3] 启动容器并生成数据（数据实时同步到宿主机）..."
mkdir -p $HOST_OUTPUT_DIR
docker run \
  --name "$CONTAINER_NAME" \
  --shm-size=2g \
  -v "./conver:/workspace/kuavo_il/v0" \
  "$IMAGE_NAME" \
  sh -c "/opt/conda/envs/myenv/bin/python /workspace/kuavo_il/kuavo/kuavo_1convert/cvt_rosbag2lerobot.py --raw_dir /workspace/kuavo_il/testbag"
docker rm "$CONTAINER_NAME" >/dev/null

echo "✅数据已经保存到：./conver"