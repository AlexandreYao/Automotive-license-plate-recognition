import torch
import math
import numpy as np
import torch.nn as nn

def bbox_iou(box1, box2, center_width_height=True):
    """
    Calcule l'Intersection sur Union (IoU) de deux boîtes englobantes.
    Args:
        box1 (torch.Tensor): Tensor représentant les coordonnées de la première boîte englobante.
        box2 (torch.Tensor): Tensor représentant les coordonnées de la deuxième boîte englobante.
        center_width_height (bool): Indique si les boîtes englobantes sont représentées sous forme de (x_centre, y_centre, largeur, hauteur) ou (x1, y1, x2, y2).
    Returns:
        torch.Tensor: Valeur de l'IoU pour chaque paire de boîtes englobantes.
    """
    if center_width_height:
        # Transformation des coordonnées du centre et de la largeur en coordonnées exactes
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Récupération des coordonnées des boîtes englobantes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # Récupération des coordonnées du rectangle d'intersection
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Calcul de la surface d'intersection
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Calcul de la surface de l'union
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    # Calcul de l'IoU
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def build_targets(targets, anchors, grid_size, num_anchors=3):
    """
    Crée les cibles pour l'entraînement du détecteur YOLO.
    Args:
        targets (torch.Tensor): Tensor contenant les annotations de vérité terrain pour chaque image.
        anchors (list): Liste des ancres utilisées pour la détection.
        grid_size (int): Taille de la grille sur laquelle la carte de caractéristiques est générée.
        num_anchors (int): Nombre d'ancres utilisées pour chaque cellule de la grille.
    Returns:
        tuple: Un tuple contenant plusieurs tensors représentant les cibles pour l'entraînement.
    """
    num_batches = targets.size(0)
    mask = torch.zeros(num_batches, num_anchors, grid_size, grid_size)
    tx = torch.zeros(num_batches, num_anchors, grid_size, grid_size)
    ty = torch.zeros(num_batches, num_anchors, grid_size, grid_size)
    tw = torch.zeros(num_batches, num_anchors, grid_size, grid_size)
    th = torch.zeros(num_batches, num_anchors, grid_size, grid_size)
    tconf = torch.zeros(num_batches, num_anchors, grid_size, grid_size)
    for batch_idx in range(num_batches):
        for target_idx in range(targets.shape[1]):
            if targets[batch_idx, target_idx].sum() == 0:
                continue
            obj_center_x = targets[batch_idx, target_idx, 1] * grid_size
            obj_center_y = targets[batch_idx, target_idx, 2] * grid_size
            obj_width = targets[batch_idx, target_idx, 3] * grid_size
            obj_height = targets[batch_idx, target_idx, 4] * grid_size
            grid_x = int(obj_center_x)
            grid_y = int(obj_center_y)
            gt_box = torch.torch.FloatTensor(
                np.array([0, 0, obj_width, obj_height])
            ).unsqueeze(0)
            anchor_shapes = torch.torch.FloatTensor(
                np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1)
            )
            ious = bbox_iou(gt_box, anchor_shapes)
            best_anchor_index = np.argmax(ious)
            mask[batch_idx, best_anchor_index, grid_y, grid_x] = 1
            tx[batch_idx, best_anchor_index, grid_y, grid_x] = obj_center_x - grid_x
            ty[batch_idx, best_anchor_index, grid_y, grid_x] = obj_center_y - grid_y
            tw[batch_idx, best_anchor_index, grid_y, grid_x] = math.log(
                obj_width / anchors[best_anchor_index][0] + 1e-16
            )
            th[batch_idx, best_anchor_index, grid_y, grid_x] = math.log(
                obj_height / anchors[best_anchor_index][1] + 1e-16
            )
            tconf[batch_idx, best_anchor_index, grid_y, grid_x] = 1
    return mask, tx, ty, tw, th, tconf


def loss(input, target, anchors, inp_dim, num_anchors=3):
    num_batches = input.size(0)  # number of batches
    grid_size = input.size(2)  # number of grid size
    stride = inp_dim / grid_size
    prediction = (
        input.view(num_batches, num_anchors, 5, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
    )  # reshape the output data
    # Get outputs
    x = torch.sigmoid(prediction[..., 0])  # Center x
    y = torch.sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height
    pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
    # Calculate offsets for each grid
    grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(torch.FloatTensor)
    grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(torch.FloatTensor)
    scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
    anchor_w = scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
    anchor_h = scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))
    # Add offset and scale with anchors
    pred_boxes = torch.FloatTensor(prediction[..., :4].shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
    mask, tx, ty, tw, th, tconf = build_targets(
        target=target.cpu().data,
        anchors=scaled_anchors.cpu().data,
        grid_size=grid_size,
        num_anchors=num_anchors,
    )
    # Handle target variables
    tx, ty = tx.type(torch.FloatTensor), ty.type(torch.FloatTensor)
    tw, th = tw.type(torch.FloatTensor), th.type(torch.FloatTensor)
    tconf = tconf.type(torch.FloatTensor)
    mask = mask.type(torch.ByteTensor)
    mse_loss = nn.MSELoss(reduction="sum")  # Coordinate loss
    bce_loss = nn.BCELoss(reduction="sum")  # Confidence loss
    loss_x = mse_loss(x[mask], tx[mask])
    loss_y = mse_loss(y[mask], ty[mask])
    loss_w = mse_loss(w[mask], tw[mask])
    loss_h = mse_loss(h[mask], th[mask])
    loss_conf = bce_loss(pred_conf, tconf)
    loss = loss_x + loss_y + loss_w + loss_h + loss_conf
    return (loss, loss_x, loss_y, loss_w, loss_h, loss_conf)