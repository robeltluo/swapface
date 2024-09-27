一 insightface模型说明
1k3d68 sim：3d识别68个关键点
2d106det：2d识别106关键点
det 10g_sim：人脸框和kps关键点5点识别 10g参数
det 500m sim：人脸框和kps关键点5点识别 500m参数
genderage：识别年龄和性别
inswapper 128_sim：人脸替换模型，可以替换照片人脸
SwapperWeightDef.dat ： 初始化权重数据
w600k mbf：人脸特征识别，轻量级库
w600k_r50:人脸特征识别，重量级
相应的onnx模型见：https://download.csdn.net/download/p731heminyang/89425467

二 安装
源码编译安装 python及openssl
https://www.cnblogs.com/chuanzhang053/p/17653635.html

./config \
    --prefix=/usr/local/openssl \
    --libdir=lib \
    --openssldir=/etc/pki/tls

    make -j1 depend
make -j8
make install_sw

sudo ./config \
    --prefix=/Users/lmc/develop/software/openssl  \
    --libdir=lib \
    --openssldir=/etc/ssl

export PATH=/usr/local/openssl/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/openssl/lib:\$LD_LIBRARY_PATH
source /etc/profile.d/openssl.sh

export CFLAGS="-I/usr/local/openssl/include"
export LDFLAGS="-L/usr/local/openssl/lib -lssl -lcrypto"
export CPPFLAGS="-I /usr/local/openssl/include"

./configure \
    --with-openssl=/usr/local/openssl \
    --with-openssl-rpath=auto \
    --prefix=/usr/local/python-3.10.0 \
    --enable-optimizations --with-zlib

export CFLAGS="-I/Users/lmc/develop/software/openssl/include"
export LDFLAGS="-L/Users/lmc/develop/software/openssl/lib -lssl -lcrypto"
export CPPFLAGS="-I /Users/lmc/develop/software/openssl/include"

./configure \
    --with-openssl=/Users/lmc/develop/software/openssl \
    --with-openssl-rpath=auto \
    --prefix=/Users/lmc/develop/software/python-3.10.0 \
    --enable-optimizations --with-zlib

brew install make cmake gcc bzip2-devel libffi-devel zlib-devel tk-devel readline-devel  gdbm-devel sqlite-devel tkinter
    yum install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

    yum install -y make cmake gcc bzip2-devel libffi-devel zlib-devel tk-devel readline-devel  sqlite-devel

./configure --prefix=/usr/local/ffmpeg --enable-openssl --disable-x86asm


三 执行

ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
xhost + $ip
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  -v ~/develop/test:/test  lmc/deep-live-cam:1.2 bash /test/run.sh

docker run -it  -v ~/develop/00_workspace_idea/00_ai/deep-live-cam:/deep-live-cam -v ~/develop/test:/test -v ~/develop/test/insightface:/root/.insightface -v ~/develop/test/opennsfw2:/root/.opennsfw2 lmc/deep-live-cam:1.2

第一次执行，下载模型：
[root@d216730420c2 test]# rm -rf /test/temp/*; python /deep-live-cam/run.py  -s /test/fbb.jpg -t /test/video2.mp4 -o /test
download_path: /root/.insightface/models/buffalo_l
Downloading /root/.insightface/models/buffalo_l.zip from https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip...
100%|██████████████████████████████████████████████████████████████| 281857/281857 [03:00<00:00, 1558.18KB/s]
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: /root/.insightface/models/buffalo_l/1k3d68.onnx landmark_3d_68 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: /root/.insightface/models/buffalo_l/2d106det.onnx landmark_2d_106 ['None', 3, 192, 192] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: /root/.insightface/models/buffalo_l/det_10g.onnx detection [1, 3, '?', '?'] 127.5 128.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: /root/.insightface/models/buffalo_l/genderage.onnx genderage ['None', 3, 96, 96] 0.0 1.0
Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
find model: /root/.insightface/models/buffalo_l/w600k_r50.onnx recognition ['None', 3, 112, 112] 127.5 127.5
set det-size: (640, 640)
Pre-trained weights will be downloaded.
Downloading...
From: https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5
To: /root/.opennsfw2/weights/open_nsfw_weights.h5
100%|███████████████████████████████████████████████████████████████████| 24.2M/24.2M [00:14<00:00, 1.67MB/s]
100%|█████████████████████████████████████████████████████████████████████| 271/271 [00:02<00:00, 123.81it/s]
[DLC.CORE] Creating temp resources...
[DLC.CORE] Extracting frames...
[DLC.FACE-SWAPPER] Progressing...


四 视频处理
切分视频
ffmpeg -i video1.mp4 -ss 00:00:03 -t 00:00:06 video2.mp4    第3秒开始裁剪，裁剪到第6秒

ffmpeg -i video1.mp4 -ss 00:00:00 -t 00:00:05 -c:v copy -c:a copy video2.mp4

ffmpeg -i video1.mp4 -ss 00:00:00 -t 00:00:05 -acodec copy -vcodec copy video2.mp4

播放视频
ffplay -autoexit input.mp4

提取视频帧
ffmpeg -hide_banner -hwaccel auto -loglevel error -i video2.mp4 -pix_fmt rgb24 tmp-frame/%04d.png

ffmpeg -hide_banner -hwaccel auto -loglevel error -r 30.0 -i /test/tmp-frame/%04d.png -c:v libx264 -crf 18 -pix_fmt yuv420p -vf colorspace=bt709:iall=bt601-6-625:fast=1 -y /test/video2.mp4

