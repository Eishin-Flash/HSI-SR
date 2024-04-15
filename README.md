# 3D-VQGAN for HSI-SR

## TimeLine

- **2024.04.15 :** create project

## instructions

- **create env :**
```
conda create -n hsi_sr python=3.8 -y
conda activate hsi_sr
cd HSI-SR
pip install -r requirements.txt
```

### Stage one(self-restruction)
- **training :**
```
cd train_code
python train_vqencoder_s1.py --batch_size 10
```

### Stage two(code-prediction)
```
update soon
```

## Experiments

- **2024.04.15 :** 

实验内容1：在3D-VQGAN自重构训练中加入感知损失（采用lpips loss，详情见train_code/utils.py）

有时间可以做个对比实验（加入/不加入）

直接输入：
```
cd train_code
python train_vqencoder_s1.py 
```
不加入感知损失：
```
cd train_code
python train_vqencoder_s1.py --use_per 0
```
实验内容2：增大模型

将train_code/architecture/Spectral_VQGAN.py line280中的nf调为5(6M模型的nf=3)
然后输入：
```
cd train_code
python train_vqencoder_s1.py --batch_size 5

```
多卡训练应该需要设置--gpu_id

实验的epoch设置为300，按MST++原项目的逻辑，一个epoch的iter是手动设定的，此处设定为10000，实际上相当于跑了大约1.5*batchsize*num_gpu个epoch
