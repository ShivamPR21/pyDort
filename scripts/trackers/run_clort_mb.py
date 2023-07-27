import os
from typing import List

import hydra
import numpy as np
import torch
import wandb
from clort import MemoryBank, MemoryBankInfer
from clort.data import ArgoCL
from clort.model import CLModel
from omegaconf import DictConfig
from tqdm import tqdm

from pyDort.helpers import flatten_cfg


@hydra.main(version_base=None, config_path="../conf", config_name="mb_config")
def run_tracker(cfg: DictConfig) -> None:
    wandb.login()

    run = wandb.init(
        project=cfg.wb.project,

        config=flatten_cfg(cfg)
    )

    dataset = ArgoCL(cfg.data.data_dir,
                    temporal_horizon=1,
                    temporal_overlap=0,
                    max_objects=cfg.data.max_objects,
                    target_cls=cfg.data.target_cls,
                    detection_score_threshold=cfg.data.det_score,
                    splits=list(cfg.data.split),
                    distance_threshold=cfg.data.distance_threshold,
                    img_size=tuple(cfg.data.img_shape),
                    point_cloud_size=list(cfg.data.pcl_quant),
                    point_cloud_scaling=1.,
                    in_global_frame=cfg.data.global_frame,
                    pivot_to_first_frame=cfg.data.pivot_to_first_frame,
                    image=cfg.data.imgs, pcl=cfg.data.pcl, bbox=cfg.data.bbox_aug,
                    vision_transform=None, # type: ignore
                    pcl_transform=None,
                    random_miss=cfg.data.random_miss)

    assert(dataset.pc_scale == 1.)

    appearance_model = CLModel(mv_backbone=cfg.am.mv_backbone,
                               mv_features=cfg.am.mv_features,
                               mv_xo=cfg.am.mv_xo,
                               pc_features=cfg.am.pc_features,
                               bbox_aug=cfg.am.bbox_aug,
                               pc_xo=cfg.am.pc_xo,
                               mm_features=cfg.am.mm_features,
                               mm_xo=cfg.am.mm_xo,
                               mmc_features=cfg.am.mmc_features,
                               mode = 'infer')

    ckpt = torch.load(wandb.restore(name=cfg.am.model_file, run_path=cfg.am.run_path, replace=True).name)
    print(f'{appearance_model.load_state_dict(ckpt["enc"]) = }')
    model_device = cfg.am.device

    appearance_model = appearance_model.to(model_device)
    appearance_model.eval()

    cur_log = None
    tracker = None

    assert(appearance_model.out_dim is not None)
    tracker_infer = MemoryBankInfer(dataset.n_tracks, N=appearance_model.out_dim,
                                    alpha_threshold=cfg.tracker.alpha_t,
                              beta_threshold=cfg.tracker.beta_t, Q = cfg.tracker.Q)
    tracker = MemoryBank(dataset.n_tracks, N=appearance_model.out_dim, Q = cfg.tracker.Q,
                         alpha=np.array(cfg.tracker.track_center_momentum, dtype=np.float32))

    for i in (run := tqdm(range(dataset.N), total=dataset.N)):
        data = dataset[i]

        rd_i, _ = dataset.get_reduced_index(i) # type: ignore
        log_id : str = dataset.log_files[rd_i].name # type: ignore

        pcls, pcls_sz, imgs, imgs_sz, bboxs, track_idxs, _cls_idxs, frame_sz, pivot = data
        cur_log_track_idxs = list(dataset.tracks[log_id].values())
        local_track_idxs = torch.tensor(get_relative_index(track_idxs.tolist(), cur_log_track_idxs), dtype=torch.long)
        cur_log_track_idxs = torch.tensor(cur_log_track_idxs, dtype=torch.long)

        if (len(track_idxs) == 0):
            continue

        # Tracking start

        log_id = log_id.split("_")[-1]
        if log_id != cur_log:
            cur_log = log_id

        run.set_description(f'Log Id: {cur_log}')

        encoding = None
        pivot = 0 if pivot is None else torch.from_numpy(pivot)
        if (len(track_idxs) != 0):
            pcls = pcls.to(model_device) if isinstance(pcls, torch.Tensor) else pcls
            imgs = imgs.to(model_device) if isinstance(imgs, torch.Tensor) else imgs
            bboxs = bboxs.to(model_device) if isinstance(bboxs, torch.Tensor) else bboxs
            pivot = pivot.to(model_device) if isinstance(pivot, torch.Tensor) else pivot

            pcl_scale = float(cfg.data.pcl_scale)
            mv_e, pc_e, mm_e, mmc_e = appearance_model((pcls - pivot)/pcl_scale, pcls_sz, imgs, imgs_sz, (bboxs - pivot)/pcl_scale, frame_sz)

            encoding = None
            if mmc_e is not None:
                encoding = mmc_e
            elif mm_e is not None:
                encoding = mm_e
            elif pc_e is not None:
                encoding = pc_e
            elif mv_e is not None:
                encoding = mv_e
            else:
                raise NotImplementedError("Encoder resolution failed.")

            encoding = encoding.detach().cpu().numpy() # go to cpu for encoding
        else:
            continue

        bboxs = bboxs.detach().cpu().numpy()

        assert(not np.any(np.isnan(encoding)))
        assert(not np.any(np.isnan(bboxs)))

        # assert(encoding is not None)
        assert(tracker is not None)

        infer_acc = evaluate(memory=tracker_infer.get_reprs(cur_log_track_idxs),
                            repr=encoding,
                            track_idxs=local_track_idxs,
                            topk=[1, 2, 3, 4, 5])
        acc = evaluate(memory=tracker.get_reprs(cur_log_track_idxs),
                        repr=encoding,
                        track_idxs=local_track_idxs,
                        topk=[1, 2, 3, 4, 5])

        tracker_infer.update(encoding, track_idxs)
        tracker.update(encoding, track_idxs)

        df = {"Idx": i}
        df.update({f'Accuracy% (Inference Memory Bank) : Top {k}': acc_*100. for k, acc_ in zip([1, 2, 3, 4, 5], infer_acc, strict=True)})
        df.update({f'XE_acc_Top_{k}': acc_*100. for k, acc_ in zip([1, 2, 3, 4, 5], acc, strict=True)})
        wandb.log(df)

    tracker_store_path = os.path.join(run.dir, 'tracker.pth')
    torch.save({"mb": tracker.state_dict(),
                "mb_infer": tracker_infer.state_dict()}, tracker_store_path)

    wandb.save(tracker_store_path)

def evaluate(memory: torch.Tensor, repr: torch.Tensor,
             # trunk-ignore(ruff/B006)
             track_idxs: torch.Tensor, topk:List[int] = [1, 2, 5]) -> float:
    # memory [n_tracks, Q, N]
    # reprs [n, N]

    n, N = repr.size()
    n_tracks, Q, _ = memory.size()

    track_idxs = track_idxs.view(-1, 1) # [n, 1]

    sim = repr @ memory.permute(2, 0, 1) # [n, n_tracks, Q]
    sim = sim.max(dim=2).values # [n, n_tracks]

    pred = torch.topk(sim, np.max(topk), dim=1, largest=True, sorted=True).indices # [n, max(topk)]

    target = torch.zeros((n, n_tracks), dtype=torch.long).scatter(1, track_idxs, 1)

    ret = []
    for k in topk:
        correct = target * torch.zeros_like(target).scatter(1, pred[:, :k], 1)
        ret.append((correct.sum()/target.sum()).item())

    return ret

def get_relative_index(source: List[int], target: List[int]) -> List[int]:
    rel_idx = []
    for s in source:
        rel_idx.append(target.index(s))

    return rel_idx

if __name__ == "__main__":
    run_tracker()
