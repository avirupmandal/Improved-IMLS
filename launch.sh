conda env create --file environment.yml

# 3DGS initialization
cd gaussian-splatting
python train.py --eval --white_background --resolution 2 --expname lego    -s $data_path

# train
cd ..
python train.py --eval --white_background --SSAA 1 --resolution 1 --expname lego    --meshscale 2.1  --start_checkpoint_ply $ckpt_point_cloud_path   -s  $data_path     
 