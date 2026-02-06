import numpy as np
import torch
from torch.nn import functional as F


def create_anchor_mask(indices, batch_size):
    zeros = np.zeros((batch_size, batch_size, batch_size))
    zeros[indices, :, :] = 1

    return torch.tensor(zeros, dtype=torch.bool)


def create_positive_mask(indices, batch_size):
    zeros = np.zeros((batch_size, batch_size, batch_size))
    zeros[:, indices, :] = 1

    return torch.tensor(zeros, dtype=torch.bool)


def create_negative_mask(indices, batch_size):
    zeros = np.zeros((batch_size, batch_size, batch_size))
    zeros[:, :, indices] = 1

    return torch.tensor(zeros, dtype=torch.bool)


def select_all_anchors(batch_size):
    return create_anchor_mask(range(batch_size), batch_size)


def select_random_anchors(batch_size, num_anchors):
    anchor_indices = np.random.randint(
        low=0, high=batch_size, size=num_anchors)
    return create_anchor_mask(anchor_indices, batch_size)


def select_diverse_anchors(distances, num_anchors):
    # Get the numpy version, so it can be used more freely.
    distances = distances.cpu().detach().numpy()

    batch_size = len(distances)
    anchor_indices = np.zeros(num_anchors, dtype=int)

    # Pick start anchor randomly.
    first = np.random.randint(low=0, high=batch_size, size=1)
    anchor_indices[0] = first

    # Iterate over rest and find most diverse one.
    for i in range(1, num_anchors):
        local_max = np.zeros(batch_size)

        # Get the already selected anchors.
        selected_anchors = anchor_indices[:i]

        for j in range(batch_size):
            # Skip already selected anchors.
            if j in selected_anchors:
                continue

            # Get the maximum distance from all already selected anchors.
            local_max[j] = distances[j, selected_anchors].max()

        # Add the most diverse anchor to the selection.
        anchor_indices[i] = np.argmax(local_max)

    return create_anchor_mask(anchor_indices, batch_size)


def select_all_valid_triplets(label_distances):
    positive = label_distances.unsqueeze(2)
    negative = label_distances.unsqueeze(0)
    return positive < negative


def select_triplets_smartly(label_distances, feature_distances, beta, gamma, num_triplets):
    batch_size = label_distances.shape[0]

    norm_feat_dist = feature_distances / torch.max(feature_distances)

    negative_qualities = beta * label_distances - (1.0 - beta) * norm_feat_dist
    positive_qualities = beta * -label_distances + (1.0 - beta) * norm_feat_dist

    if gamma == 1.0:  # i.e. no diversity
        values, _ = torch.topk(positive_qualities, k=num_triplets, dim=1)
        maximums = values[:, num_triplets - 1]
        positives = torch.ge(positive_qualities, torch.unsqueeze(maximums, 1))

        values, _ = torch.topk(negative_qualities, k=num_triplets, dim=1)
        maximums = values[:, num_triplets - 1]
        negatives = torch.ge(negative_qualities, torch.unsqueeze(maximums, 1))

        return torch.logical_and(positives.unsqueeze(2), negatives.unsqueeze(1))

    else:
        # Initialize the positive mask.
        positive_mask = F.one_hot(torch.argmax(positive_qualities, dim=1), batch_size).bool()

        # Initialize the negative mask.
        negative_mask = F.one_hot(torch.argmax(negative_qualities, dim=1), batch_size).bool()

        for _ in range(num_triplets - 1):
            # Get positive samples with highest diversity.
            local_max = torch.max(torch.mul(positive_mask.float(), norm_feat_dist), dim=1)[0]

            final_quality = gamma * positive_qualities + (1.0 - gamma) * local_max.unsqueeze(1)
            masked_final_quality = final_quality - positive_mask.float() * torch.max(final_quality)

            positive_mask_diverse = F.one_hot(torch.argmax(masked_final_quality, dim=1), batch_size).bool()
            positive_mask = torch.logical_or(positive_mask, positive_mask_diverse)

            # Get negative samples with highest diversity.
            local_max = torch.max(torch.mul(negative_mask.float(), norm_feat_dist), dim=1)[0]

            final_quality = gamma * negative_qualities + (1.0 - gamma) * local_max.unsqueeze(1)
            masked_final_quality = final_quality - negative_mask.float() * torch.max(final_quality)

            negative_mask_diverse = F.one_hot(torch.argmax(masked_final_quality, dim=1), batch_size).bool()

            negative_mask = torch.logical_or(negative_mask, negative_mask_diverse)

        return torch.logical_and(positive_mask.unsqueeze(2), negative_mask.unsqueeze(1))