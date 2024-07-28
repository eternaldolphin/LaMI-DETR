from detrex.config import get_config
from .models.dino_convnextl import model

model.vlm_query_path = "dataset/metadata/lvis_visual_desc_confuse_lvis_convnextl.npy"
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes='dataset/VisualGenome/lvis_v1_seen_classes.json'
model.all_classes='dataset/VisualGenome/lvis_v1_all_classes.json'
model.vlm_temperature=100.0
model.alpha=0
model.beta=0.25

# get default config
dataloader = get_config("common/data/vg_train_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/vg_schedule.py").lr_multiplier_12ep_32bs_warmup
train = get_config("common/train.py").train


# modify training config
# train.init_checkpoint = "./pretrained_models/idow_convnext_large_obj365_12ep.pth" # for train
train.init_checkpoint = "./pretrained_models/idow_convnext_large_12ep_vg/model_final.pth" # for inference
train.output_dir = "./output/idow_convnext_large_12ep_vg"

# max training iterations
train.max_iter = 72000

# run evaluation every 5000 iters
train.eval_period = 6000

# log training infomation every 20 iters
train.log_period = 200

# save checkpoint every 5000 iters
train.checkpointer.period = 6000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

model.num_classes = 8212
model.query_path = "dataset/metadata/concept_dict_visual_desc_convnextl.npy"
model.eval_query_path = "dataset/metadata/lvis_visual_desc_convnextl.npy"

model.use_fed_loss = True
model.cluster_fed_loss = True
model.cluster_label_path = 'dataset/cluster/vg_cluster_256.npy'
model.cat_freq_path ="dataset/VisualGenome/vg_filter_rare_cat_info.json"
model.fed_loss_num_cat=700
model.select_box_nums_for_evaluation = 300

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 8

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 32

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
dataloader.test.dataset.names = "lvis_v1_val"
