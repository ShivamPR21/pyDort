# pyDort
Python Deep Online Realtime Tracking

### Testing scripts

1. Simple IoU, Proximity, and Orientation based data association

    ```sh
    python scripts/run_tracker_filterpy.py --dets_dataroot $HOME/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/ --raw_data_dir $HOME/Research/datasets/argoverse-tracking/train1/ --min_hits 1 --max_age 5 --tracks_dump_dir results/tracks_dump_exp1_simple_tracker --config_file conf/conf-cv-cv.json --target_cls PEDESTRIAN > results/out1.txt
    ```

2. Image based deep metric with CNNs

    ```sh
    python scripts/run_tracker_resnet_filterpy.py --dets_dataroot $HOME/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/ --raw_data_dir $HOME/Research/datasets/argoverse-tracking/train1/ --min_hits 1 --max_age 5 --tracks_dump_dir results/tracks_dump_exp2_convnext_tiny_tracker --config_file conf/conf-cv-cv.json --target_cls PEDESTRIAN --gpu --model convnext_tiny --agr avg --chunk_size 2 > results/out2.txt
    ```

3. Image + Point Cloud based deep metric for data association

    ```sh
    python scripts/run_tracker_multi_modal_filterpy.py --dets_dataroot $HOME/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/ --raw_data_dir $HOME/Research/datasets/argoverse-tracking/train1/ --min_hits 1 --max_age 5 --tracks_dump_dir results/tracks_dump_exp3_mm_tracker --config_file conf/conf-cv-cv.json --target_cls PEDESTRIAN --im_gpu --im_model convnext_tiny --pcd_model dgcnn1024 --agr avg --im_chunk_size 5 --pcd_chunk_size 5 > results/out3.txt
    ```

4. Image based deep metric with Contrastive CNNs

    ```sh
    python scripts/run_tracker_contrastive_filterpy.py --dets_dataroot $HOME/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/ --raw_data_dir $HOME/Research/datasets/argoverse-tracking/train1/ --min_hits 1 --max_age 5 --tracks_dump_dir results/tracks_dump_exp3_resnet18_cl_tracker --config_file conf/conf-cv-cv.json --target_cls PEDESTRIAN --gpu --model resnet18 --agr avg --chunk_size 2 > results/out4.txt
    ```

5. Image(Contrastive) + Point Cloud based deep metric for data association

    ```sh
    python scripts/run_tracker_contrastive_multi_modal_filterpy.py --dets_dataroot $HOME/Research/datasets/argoverse-tracking/test/argoverse_detections_2020/training/ --raw_data_dir $HOME/Research/datasets/argoverse-tracking/train1/ --min_hits 1 --max_age 5 --tracks_dump_dir results/tracks_dump_exp5_mm_cl_tracker --config_file conf/conf-cv-cv.json --target_cls PEDESTRIAN --im_gpu --im_model resnet18 --pcd_model dgcnn1024 --agr avg --im_chunk_size 5 --pcd_chunk_size 5 > results/out5.txt
    ```
