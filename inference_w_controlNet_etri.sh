CUDA_VISIBLE_DEVICES=6 python inference_w_controlNet_etri.py --config configs/inference/inference_pose2img.yaml \
                                            --ref_path "/home/cvlab15/project/jeeyoung/etri_video/C3_frames/2024_08_28_09_20_56/images/frame_8.jpg"\
                                           --seg_path "/home/cvlab15/project/jeeyoung/etri_video/C3_frames/2024_08_28_09_20_56/semantic/frame_8.jpg" \
                                           --output_dir ./outputs_2 \
                                           --width 512 \
                                           --height 512 \
                                           --seed 42 \
                                           --reference_unet_path "/home/cvlab15/project/jeeyoung/etri_overall_architecture/stage1/reference_unet-37704.pth" \
                                           --pose_guider_path "/home/cvlab15/project/jeeyoung/etri_overall_architecture/stage1/pose_guider-37704.pth" \
                                           --denoising_unet_path "/home/cvlab15/project/jeeyoung/etri_overall_architecture/stage1/denoising_unet-37704.pth" \