python3 ./train_ag.py --arch APDNet --batch_size 16 --gpu '7' --nepoch 1000 \
      --train_ps 256 \
      --train_gt_dir /data02/chikaichen/all-in-one/Denoising/shuju/OPTIMAL-31/train/gt/  \
      --train_input_dir /data02/chikaichen/all-in-one/Denoising/shuju/OPTIMAL-31/train/input/ \
      --val_gt_dir /data02/chikaichen/all-in-one/Denoising/shuju/OPTIMAL-31/test/gt/ \
      --val_input_dir /data02/chikaichen/all-in-one/Denoising/shuju/OPTIMAL-31/test/input/ \
      --embed_dim 64 --warmup --checkpoint 500 \
      --env APDNetdim64_G15_1 --noiseL 15 --lr_initial 0.0001
