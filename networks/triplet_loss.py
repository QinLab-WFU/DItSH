import networks.triplet_selection as triplet_selection
from torch import nn
import torch
from torch.nn import functional as F


def _get_anchor_selection_function(fd, num_anchors):
    return triplet_selection.select_diverse_anchors(fd, num_anchors)


def _get_default_triplet_strategy(ld):
    return triplet_selection.select_all_valid_triplets(ld)


def _get_triplet_selection_function(feature_distances, label_distances):

    beta, gamma = 0.5, 0.2
    num_elements = 30

    return triplet_selection.select_triplets_smartly(
                            label_distances, feature_distances, beta, gamma, num_elements)


def select_distinct_triplets(batch_size):
    indices_equal = torch.eye(batch_size, dtype=torch.bool)
    indices_not_equal = torch.logical_not(indices_equal)

    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    return torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


def select(features, feature_distances, label_distances):

    batch_size = len(features)

    base_mask = select_distinct_triplets(batch_size)
    base_mask = base_mask.to(0)
    anchor_mask = _get_anchor_selection_function(feature_distances, 30)
    anchor_mask = anchor_mask.to(0)
    triplet_mask = _get_triplet_selection_function(feature_distances, label_distances)
    triplet_mask = triplet_mask.to(0)

    return torch.logical_and(base_mask, torch.logical_and(anchor_mask, triplet_mask))


def calculate_pairwise_feature_distances(source, target):
    dot_product = torch.matmul(source, target.t())
    square_norm = torch.diagonal(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * \
        dot_product + square_norm.unsqueeze(1)
    distances = torch.maximum(distances, torch.tensor(0.0))

    mask = torch.eq(distances, torch.tensor(0.0)).float()
    distances = distances + mask * 1e-16
    distances = torch.sqrt(distances)
    distances = distances * (1.0 - mask)

    return distances * (1.0 - mask)


def _cos_distance(source, target):
    """
    Compulate the 2D matrix of cosine distance between all the source and target vectors.
    :param source: tensor of shape (batch_size, embed_dim)
    :param target: tensor of shape (*, embed_dim)
    :return: tensor of shape (batch_size, batch_size)
    """

    cos_sim = F.cosine_similarity(source.unsqueeze(1), target, dim=-1)

    # put everything >= 0
    distances = torch.clamp(1 - cos_sim, 0)

    return distances


def calculate_pairwise_label_distances(labels):
    labels = labels.float()
    dot_product = torch.matmul(labels, labels.t())
    square_norm = torch.sqrt(torch.diagonal(dot_product))
    norms = square_norm.unsqueeze(0) * square_norm.unsqueeze(1)

    result = torch.minimum(torch.tensor(1.0), dot_product / norms)
    return 1.0 - result


class TripletLoss(nn.Module):
    """
    A loss to judge similarity based on a triplet consisting of three input
    elements, called anchor, positive, and negative.
    """

    def __init__(self, margin=0.5):
        """
        Creates a new loss instance.

        Args:
            margin (float): the distance that is forced between the positive and negative distance.
        """

        super(TripletLoss, self).__init__()

        if margin < 0:
            raise ValueError('Margin has to be >= 0.')

        self.margin = margin

    def forward(self, label, source, target):
        feature_distances = _cos_distance(source, target)
        label_distances = calculate_pairwise_label_distances(label.float())

        anchor_positive_dist = feature_distances.unsqueeze(2)
        anchor_negative_dist = feature_distances.unsqueeze(1)

        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        mask = select(source, feature_distances, label_distances)
        # mask = mask.cpu()
        mask = mask.to('cuda:0')
        triplet_loss = mask * triplet_loss

        triplet_loss = triplet_loss.clamp(0)

        valid_triplets = mask.float()
        num_positive_triplets = torch.sum(valid_triplets)


        return torch.sum(triplet_loss) / (num_positive_triplets + 1e-4)