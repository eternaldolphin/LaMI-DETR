import json
import torch

def load_class_freq(
    path='datasets/metadata/lvis_v1_train_cat_info.json', freq_weight=0.5):
    cat_info = json.load(open(path, 'r'))
    cat_info = torch.tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    cat_info = cat_info.float()
    freq_weight = cat_info ** freq_weight
    return freq_weight

def get_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared

def get_cluster_fed_loss_inds(gt_classes, num_sample_cats, C, weight=None, cluster_label=None):
    appeared = torch.unique(gt_classes) # C'
    prob = appeared.new_ones(C + 1).float()
    prob[-1] = 0
    if cluster_label is not None:
        cluster_label = torch.tensor(cluster_label)# [8212]
        gt_classes_cluster = cluster_label[appeared]# [58]
        appeared_cluster = torch.unique(gt_classes_cluster)# [38]
        same_cluster_class = torch.nonzero(torch.isin(cluster_label, appeared_cluster)).squeeze()
        other_cluster_class = torch.nonzero(~torch.isin(cluster_label, appeared_cluster)).squeeze()
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.float().clone()
        prob[appeared] = 0
        if cluster_label is not None:
            prob[same_cluster_class] = 0# 只取不同cluster
        more_appeared = torch.multinomial(
            prob, num_sample_cats - len(appeared),
            replacement=False)
        appeared = torch.cat([appeared, more_appeared])
    return appeared