import glob
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from imageio import imread
import pyiqa
from pyiqa import create_metric



label_path = 'path for your label images'
image_path = 'path for your generated images'


label_list = os.listdir(label_path)
image_list = os.listdir(image_path)

#-----------------ssim & psnr-----------------#
sum_ssim = 0
sum_psnr = 0

for i in range(len(label_list)):
    label = imread((label_path+ label_list[i]))
    image = imread((image_path+ image_list[i]))
    sum_ssim += ssim(label, image,channel_axis=-1)
    #sum_ssim += ssim(label, image,multichannel=True)
    sum_psnr += psnr(label, image)


if os.path.isfile(label_path):
    input_paths = [label_path]
    if image_path is not None:
        ref_paths = [image_path]
else:
    input_paths = sorted(glob.glob(os.path.join(label_path, '*')))
    if image_path is not None:
        ref_paths = sorted(glob.glob(os.path.join(image_path, '*')))

#-----------------lpips-----------------#
avg_lpips = 0
iqa_model = create_metric('lpips', metric_mode='FR')
metric_mode = iqa_model.metric_mode
test_img_num = len(input_paths)
for idx, img_path in enumerate(input_paths):
    img_name = os.path.basename(img_path)
    if metric_mode == 'FR':
        ref_img_path = ref_paths[idx]
    else:
        ref_img_path = None

    lpips_score = iqa_model(img_path, ref_img_path).cpu().item()
    avg_lpips += lpips_score

#-----------------fid-----------------#
fid_metric = pyiqa.create_metric('fid')
fid_score = fid_metric(label_path, image_path)


print("SSIM:",round(sum_ssim/len(label_list),4))
print("PSNR:",round(sum_psnr/len(label_list),4))
print("LPIPS:",round(avg_lpips/test_img_num,4))
print("FID:",round(fid_score,4))
