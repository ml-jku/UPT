import torch
from torch.nn.functional import softmax


# Method to calculate the invariance of the latent space representations
# Eq.6 in https://openreview.net/pdf?id=SCD0hn3kMHw
# features_no_aug.shape = (N,d)
# feature_dict_augs: each key represents an original sample -> number of keys = N
#                    each key contains an array with features with shape (L,d), 
#                    where L is the number of augmentations
def calc_geed_cosine(features_no_aug, feature_dict_augs):
    d = features_no_aug.shape[0]
    # Calculate the normalization constant M
    features_no_aug = features_no_aug / features_no_aug.norm(dim=1, keepdim=True)
    sim = features_no_aug @ features_no_aug.T
    m_norm = sim.triu(diagonal=1).sum() * 2 / (d ** 2 - d)
    avg_sim_in_aug = []
    for key in feature_dict_augs:
        features = feature_dict_augs[key]
        features = features / features.norm(dim=1, keepdim=True)
        avg_feature = features.mean(dim=0)
        avg_feature = avg_feature / avg_feature.norm()
        avg_sim_in_aug.append((features @ avg_feature).mean().item())
    geed_unnorm = sum(avg_sim_in_aug) / len(avg_sim_in_aug)
    geed_norm = geed_unnorm / m_norm.item()

    return {'geed_normalized': geed_norm,
            'geed_unnormalized': geed_unnorm,
            'norm': m_norm.item()}


# Method to calculate the effective invariance of a classifier without labels
# Eq.1 in https://arxiv.org/abs/2207.07065
# logits_no_augs.shape = (N,n_classes)
# logits_with_augs.shape = (N,n_classes)
def calc_ei(logits_no_augs, logits_with_augs):
    probs_no_augs = softmax(logits_no_augs, dim=1)
    probs_with_augs = softmax(logits_with_augs, dim=1)
    probs_no_augs_correct_idx = probs_no_augs.argmax(dim=1)
    probs_with_augs_correct_idx = probs_with_augs.argmax(dim=1)
    mask = probs_no_augs_correct_idx == probs_with_augs_correct_idx
    combined_probs = (probs_no_augs * probs_with_augs)
    all_ei = torch.sqrt(combined_probs[torch.arange(len(probs_no_augs)), probs_no_augs_correct_idx]) * mask
    ei = all_ei.mean().item()
    return ei
