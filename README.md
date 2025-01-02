# **High-resolution Medical Image Translation via Patch Alignment-based Bidirectional Contrastive Learning (MICCAI 2024)** [[Paper Link](https://link.springer.com/chapter/10.1007/978-3-031-72083-3_17)]

Thank you for your attention. This is the codebase for Patch alignment-based Paired image-to-image Translation (PPT). Please feel free to raise any question.



## Environment

dominate==2.6.0 <br>
h5py==3.7.0 <br>
numpy==1.21.5 <br>
opencv_python==4.6.0.66 <br>
packaging==21.3 <br>
Pillow==10.0.0 <br>
torch==1.11.0 <br>
torchvision==0.12.0 <br>
visdom==0.2.3 <br>



## Dataset  directory

Put the paired images into the below format, where A is your source images ( H&E stained images) and B is your target images (IHC stained images). 

```
├─Dataset
│  ├─trainA
│  ├─trainB
│  ├─testA
│  └─testB
```



## Training

python train.py --dataroot /home/user/ppt/dataset/ --name test



## Testing

python test.py --dataroot /home/user/ppt/dataset/ --name test --phase test --how_many testing_size --serial_batches --which_epoch latest



## Evaluation

Our code for metrics calculation is provided in evaluation.py



## Dataset

* [BAIDU Disk](https://pan.baidu.com/s/1kQ8PdEbWVcuqwEtc3T9avg?pwd=dsgh) Passcode:dsgh

* [Huggingface](https://huggingface.co/datasets/wzhang472/HIT/tree/main/HIT)

If you have any trouble accessing our dataset, feel free to let us know. 



## Citation

 Please feel free to cite us if our work can be helpful for your study. Thank you.

```
@inproceedings{zhang2024ppt,
  title={High-Resolution Medical Image Translation via Patch Alignment-Based Bidirectional Contrastive Learning},
  author={Zhang, Wei and Hui, Tik Ho and Tse, Pui Ying and Hill, Fraser and Lau, Condon and Li, Xinyue},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={178--188},
  year={2024},
  organization={Springer}
}
```



## Acknowledgement

Our code uses libraries from [Pix2pixHD](https://github.com/NVIDIA/pix2pixHD), [CUT](https://github.com/taesungp/contrastive-unpaired-translation) and [ASP](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE).
