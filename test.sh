python3 ./test_pad_ad.py --arch APDNet --batch_size 1 --gpu '7' \
    --input_dir /data02/chikaichen/all-in-one/Denoising/shuju/OPTIMAL-31/test/input/ \
    --gt_dir /data02/chikaichen/all-in-one/Denoising/shuju/OPTIMAL-31/test/gt/ \
    --save_in /data02/chikaichen/all-in-one/Denoising/APD-Nets/output/ \
    --result_dir /data02/chikaichen/all-in-one/Denoising/APD-Nets/result/ \
    --weights /data02/chikaichen/all-in-one/Denoising/APD-Nets/log/APDNetAPDNetdim64_G15_1/models/model_best.pth \
    --embed_dim 64 --val_ps 64 --noiseL 15
