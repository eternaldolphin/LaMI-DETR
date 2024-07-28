from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler


def default_obj365_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
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
    total_steps_16bs = epochs * 110000
    decay_steps = decay_epochs * 110000
    warmup_steps = warmup_epochs * 110000
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps_16bs],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )

def bs64_obj365_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
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
    total_steps_64bs = epochs * 27500
    decay_steps = decay_epochs * 27500
    warmup_steps = int(warmup_epochs * 27500)
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps_64bs],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_64bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )

def bs32_obj365_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
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
    total_steps_32bs = epochs * 55000
    decay_steps = decay_epochs * 55000
    warmup_steps = int(warmup_epochs * 55000)
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps_32bs],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_32bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )




# default scheduler for detr
lr_multiplier_50ep = default_obj365_scheduler(50, 40, 0)
lr_multiplier_36ep = default_obj365_scheduler(36, 30, 0)
lr_multiplier_24ep = default_obj365_scheduler(24, 20, 0)
lr_multiplier_12ep = default_obj365_scheduler(12, 11, 0)

# warmup scheduler for detr
lr_multiplier_50ep_warmup = default_obj365_scheduler(50, 40, 1e-3)
lr_multiplier_12ep_warmup = default_obj365_scheduler(12, 11, 1e-3)

lr_multiplier_50ep_warmup_bs64 = bs64_obj365_scheduler(50, 40, 1e-3)
lr_multiplier_12ep_warmup_bs64 = bs64_obj365_scheduler(12, 11, 1e-3)
lr_multiplier_24ep_warmup_bs64 = bs64_obj365_scheduler(24, 20, 1e-3)

lr_multiplier_12ep_warmup_bs32 = bs32_obj365_scheduler(12, 11, 1e-3)

