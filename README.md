# Super-Resolution of Ultra-High Magnification
## Lightweight Diffusion Model Approach
### Paper Status: Under Review

In this project, we focus on remote sensing image resolution at an ultra-high magnification factor (x8). At such high magnification, CNN-based methods provide lower perceptual image quality, while GAN-based methods show reduced authenticity and more artifacts in generated images. Despite the success of recent diffusion models like SR3, they suffer from long inference times (over 50 seconds to create a 224x224 image on a single GeForce RTX 3090 GPU). To tackle this, we develop a lightweight diffusion model called LWTDM. LWTDM uses a cross-attention-based encoder-decoder network and leverages DDIM sampling. While there is a decrease in image quality, LWTDM excels in fast generation (under 2 seconds for the same scene). Moreover, compared to other models with similar speeds, LWTDM exhibits superior performance, especially tailored for highly specific remote sensing scenarios.


Our Contributions:
1.  Introducing a lightweight model that converges rapidly.
2.  Enabling DDIM sampling support.

![LWTDM](https://github.com/Suanmd/LWTDM/blob/main/img/LWTDM.png)

*The specific model will be uploaded later.


### Environment Setup

    pip install -r requirements.txt

### Data Preparation

    python data/prepare_data.py --path path/to/data --out path/to/dataset --size 28,224

For generating downsampled data, use `generate_bicubic_img.m`

### Training

    python sr2.py -p train -c config/sr_lwtdm.json -gpu num
    python sr.py -p train -c config/sr_srddpm.json -gpu num
    python sr.py -p train -c config/sr_sr3.json -gpu num

### Testing

    python infer.py -c config/sr_lwtdm.json -gpu num
    python infer.py -c config/sr_srddpm.json -gpu num
    python infer.py -c config/sr_sr3.json -gpu num

### Evaluation
Compute PSNR, SSIM, and FID metrics.

    python run_metrics.py project_name cuda_device
 
Please replace `project_name` with the complete name found in the experiments folder, and replace `cuda_device` with a numerical value.

### Complexity Evaluation
Refer to the **cal_complex** folder for details.

------
### References

 1. [CDCR](https://github.com/Suanmd/CDCR)
 2. [ESRGAN](https://github.com/XPixelGroup/BasicSR)
 3. [SRDiff](https://github.com/LeiaLi/SRDiff)
 4. [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)
