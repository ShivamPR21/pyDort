# set -x data_partition train1
set -x raw_data_dir {$HOME}/Research/datasets/argoverse-tracking/{$data_partition}/
set -x dets_dataroot {$HOME}/Research/datasets/argoverse-tracking/{$data_partition}/
set -x out_dir {$HOME}/Research/datasets/argoverse-tracking/argov1_proc/
set -x scripts_dir {$HOME}/Research/MOT-Research/pyDort/scripts/trackers
set -x n_logs 100


mkdir -p $out_dir

python {$scripts_dir}/tracking_data_dump.py \
        --dets_dataroot $dets_dataroot \
        --raw_data_dir $raw_data_dir \
        --tracks_dump_dir $out_dir \
        --n_logs $n_logs \
        --d_part $data_partition
