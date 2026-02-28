import torch
import torch.nn as nn

class YOLO_Loss(nn.Module):
    def __init__(self, S, B, C,lambda_coord=5.0,lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, targets):

        obj_mask = targets[...,4] == 1
        noobj_mask = targets[...,4] == 0
        ignore_mask = targets[...,4] == -1

        # ---------------- BOX LOSS ----------------
        pred_xy = torch.sigmoid(preds[...,0:2])
        pred_wh = preds[...,2:4]

        target_xy = targets[...,0:2]
        target_wh = targets[...,2:4]

        xy_loss = self.mse(pred_xy, target_xy).sum(dim=-1)
        wh_loss = self.mse(pred_wh, target_wh).sum(dim=-1)

        box_loss = self.lambda_coord * (
            xy_loss[obj_mask].sum() +
            wh_loss[obj_mask].sum()
        )

        # ---------------- CONF LOSS ----------------
        pred_conf = torch.sigmoid(preds[...,4])
        target_conf = torch.clamp(targets[...,4], min=0)

        conf_loss_obj = self.mse(
            pred_conf[obj_mask],
            target_conf[obj_mask]
        ).sum()

        # IMPORTANT: exclude ignore anchors
        valid_noobj_mask = noobj_mask & (~ignore_mask)

        conf_loss_noobj = self.lambda_noobj * self.mse(
            pred_conf[valid_noobj_mask],
            target_conf[valid_noobj_mask]
        ).sum()

        conf_loss = conf_loss_obj + conf_loss_noobj

        # ---------------- CLASS LOSS ----------------
        pred_cls = torch.sigmoid(preds[...,5:5+self.C])
        target_cls = targets[...,5:5+self.C]

        cls_loss = self.mse(
            pred_cls[obj_mask],
            target_cls[obj_mask]
        ).sum()

        total_loss = box_loss + conf_loss + cls_loss

        total_loss = total_loss / preds.size(0)

        return total_loss