# IMPROVED IMLS SPLATTING

This project extends IMLS-Splatting by introducing stochastic preconditioning and exploring alternative kernel formulations, including an implicit kernel, to improve reconstruction stability and surface quality.

## Training

### Point Cloud and Normal Initialization

```bash
cd gaussian-splatting
python train.py --eval --white_background --resolution 2 --expname $exp_name -s $data_path

# for example
python train.py --eval --white_background --resolution 2 --expname lego -s ./data/nerf_synthetic/lego
```

### Run IMLS Splatting

```bash
cd ..
python train.py --eval --white_background --SSAA 1 --resolution 1 \
  --expname $exp_name \
  --meshscale $scene_scale \
  --start_checkpoint_ply $3DGS_ply_path \
  -s $data_path

# for example
python train.py --eval --white_background --SSAA 1 --resolution 1 \
  --expname lego \
  --meshscale 2.1 \
  --start_checkpoint_ply ./gaussian-splatting/output/lego/point_cloud/iteration_7000/point_cloud.ply \
  -s ./data/nerf_synthetic/lego
```


