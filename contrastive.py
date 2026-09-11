"""
contrastive.py -- signer-invariant representation learning.

Motivation
----------
A sign is the same sign whoever produces it, but the pixels are not: hand
morphology, skin tone, lighting, camera angle and signing speed all vary
between people.  A purely discriminative objective has no reason to place
two signers' renditions of the same sign near each other in feature space
-- it only has to get the argmax right -- so the representation is free to
encode signer identity alongside sign identity.  That is the mechanism
behind the demographic gaps reported for this model family (-8.9% on
elderly signers, -6.3% on darker skin tones) and behind the cross-dataset
drop.

A supervised contrastive term addresses this directly: it pulls every
same-sign sample in the batch together regardless of who produced it, and
pushes different signs apart.  Because the positives are drawn across
signers, signer identity becomes a nuisance direction that the objective
actively suppresses.

Two further reasons it fits this problem:

  * Data efficiency.  The contrastive term supplies O(B^2) training signal
    per batch from pairwise relations rather than O(B) from labels alone,
    which matters at the scale of the corpus actually available here.
  * It shapes the *representation*, which is what the collapse diagnostics
    measure, so it is directly comparable to the other interventions.

Loss
----
Supervised contrastive (Khosla et al., 2020).  With normalised embeddings
z_i, temperature t, and P(i) the same-class positives of anchor i:

    L = sum_i  -1/|P(i)| sum_{p in P(i)}
            log  exp(z_i . z_p / t) / sum_{a != i} exp(z_i . z_a / t)

Anchors with no positive in the batch contribute nothing, which matters
here: with 23 classes and a modest batch some classes appear once.  Using
two augmented views per sample guarantees every anchor at least one
positive.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection onto the unit sphere.

    The contrastive loss is applied to this projection rather than to the
    classifier features directly, which is standard: it lets the encoder
    keep information the contrastive objective would otherwise discard.
    """

    def __init__(self, in_dim, hidden=None, out_dim=64):
        super().__init__()
        hidden = hidden or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h):
        return F.normalize(self.net(h), dim=-1)


def supervised_contrastive_loss(z, labels, temperature=0.1, eps=1e-8):
    """SupCon loss.

    Parameters
    ----------
    z : (N, d) L2-normalised embeddings.  With two views per sample, pass
        both views stacked and their labels repeated.
    labels : (N,) integer class labels.

    Returns
    -------
    scalar loss.  Anchors with no same-class positive are skipped.
    """
    N = z.shape[0]
    dev = z.device
    sim = z @ z.t() / temperature

    # numerical stability: subtract the row max before exponentiating
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(N, dtype=torch.bool, device=dev)
    pos_mask = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return z.new_zeros(())

    mean_log_prob_pos = (
        (pos_mask.float() * log_prob).sum(dim=1)[valid] / n_pos[valid].float()
    )
    return -mean_log_prob_pos.mean()


@torch.no_grad()
def intra_class_compactness(z, labels):
    """Mean cosine similarity within a class minus that between classes.

    A single interpretable number for whether same-sign samples sit closer
    together than different-sign samples.  Higher is better; zero means
    the representation does not separate signs at all.
    """
    z = F.normalize(z, dim=-1)
    sim = z @ z.t()
    N = z.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    same = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye
    diff = (~same) & ~eye
    if same.sum() == 0 or diff.sum() == 0:
        return float("nan")
    return float(sim[same].mean() - sim[diff].mean())
