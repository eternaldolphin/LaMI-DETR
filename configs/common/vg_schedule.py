from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

def default_lvis_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0, batch_size=16,):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_images = 96000
    iters_per_epoch = int(total_images / batch_size) 
    total_steps = epochs * iters_per_epoch
    decay_steps = decay_epochs * iters_per_epoch
    warmup_steps = warmup_epochs * iters_per_epoch
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps,
        warmup_method="linear",
        warmup_factor=0.001,
    )


# default scheduler for detr
lr_multiplier_50ep = default_lvis_scheduler(50, 40, 0, 16)
lr_multiplier_36ep = default_lvis_scheduler(36, 30, 0, 16)
lr_multiplier_24ep = default_lvis_scheduler(24, 20, 0, 16)
lr_multiplier_12ep = default_lvis_scheduler(12, 11, 0, 16)

# warmup scheduler for detr
lr_multiplier_50ep_warmup = default_lvis_scheduler(50, 40, 1e-3, 16)
lr_multiplier_12ep_warmup = default_lvis_scheduler(12, 11, 1e-3, 16)

lr_multiplier_24ep_64bs = default_lvis_scheduler(24, 20, 0, 64)
lr_multiplier_12ep_64bs = default_lvis_scheduler(12, 11, 0, 64)

lr_multiplier_24ep_32bs = default_lvis_scheduler(24, 20, 0, 32)
lr_multiplier_12ep_32bs = default_lvis_scheduler(12, 11, 0, 32)
lr_multiplier_12ep_32bs_warmup = default_lvis_scheduler(12, 11, 1e-3, 32)
