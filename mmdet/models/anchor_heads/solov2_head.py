import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.ops import DeformConv, roi_align, ConvModule
from mmdet.core import multi_apply, bbox2roi, matrix_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob

from mmdet.core import force_fp32

INF = 1e8

from scipy import ndimage

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return 1-d

@HEADS.register_module
class SOLOV2Head(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.4,
                 num_grids=None,
                 cate_down_pos=0,
                 limit_sample_num=1,
                 with_deform=False,
                 loss_ins=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None):
        super(SOLOV2Head, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform
        self.limit_sample_num=limit_sample_num
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False  # NOTE added
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.feature_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        
        # mask feature     
        for i in range(4):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = ConvModule(
                    self.in_channels,
                    self.seg_feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=norm_cfg is None)
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.feature_convs.append(convs_per_level)
                continue
            for j in range(i):
                if j == 0:
                    if i==3:
                        in_channel = self.in_channels+2
                    else:
                        in_channel = self.in_channels
                    one_conv = ConvModule(
                        in_channel,
                        self.seg_feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=norm_cfg is None)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), one_upsample)
                    continue
                one_conv = ConvModule(
                    self.seg_feat_channels,
                    self.seg_feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=norm_cfg is None)
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)
            self.feature_convs.append(convs_per_level)
 

        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels

            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
            chn = self.in_channels if i == 0 else self.seg_feat_channels 
            self.cate_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.seg_feat_channels, 1, padding=0)
        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.cate_out_channels, 3, padding=1)
        self.solo_mask = ConvModule(
            self.seg_feat_channels, self.seg_feat_channels, 1, padding=0, norm_cfg=norm_cfg, bias=norm_cfg is None)
 
    def init_weights(self):
        #TODO: init for feat_conv
        for m in self.feature_convs:
            s=len(m)
            for i in range(s):
                if i%2 == 0:
                    normal_init(m[i].conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (feats[0].shape[-2], feats[0].shape[-3])
        kernel_pred, cate_pred = multi_apply(self.forward_single, new_feats,
                                          list(range(len(self.seg_num_grids))),
                                          eval=eval)
        # add coord for p5
        x_range = torch.linspace(-1, 1, feats[-2].shape[-1], device=feats[-2].device, dtype=feats[-2].dtype)
        y_range = torch.linspace(-1, 1, feats[-2].shape[-2], device=feats[-2].device, dtype=feats[-2].dtype)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([feats[-2].shape[0], 1, -1, -1])
        x = x.expand([feats[-2].shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        feature_add_all_level = self.feature_convs[0](feats[0]) 
        for i in range(1,3):
            feature_add_all_level = feature_add_all_level + self.feature_convs[i](feats[i])
        feature_add_all_level = feature_add_all_level + self.feature_convs[3](torch.cat([feats[3],coord_feat],1))
        
        feature_pred = self.solo_mask(feature_add_all_level)   
        N, c, h, w = feature_pred.shape
        feature_pred = feature_pred.view(-1, h, w).unsqueeze(0)
        ins_pred = []
        
        for i in range(5):
            kernel = kernel_pred[i].permute(0,2,3,1).contiguous().view(-1,c).unsqueeze(-1).unsqueeze(-1)
            ins_i = F.conv2d(feature_pred, kernel, groups=N).view(N,self.seg_num_grids[i]**2, h,w)
            if not eval:
                ins_i = F.interpolate(ins_i, size=(featmap_sizes[i][0]*2,featmap_sizes[i][1]*2), mode='bilinear')
            if eval:
                ins_i=ins_i.sigmoid()
            ins_pred.append(ins_i)
        return ins_pred, cate_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'), 
                feats[1], 
                feats[2], 
                feats[3], 
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def forward_single(self, x, idx, eval=False):
        kernel_feat = x
        cate_feat = x
        # kernel branch
        # concat coord
 
        x_range = torch.linspace(-1, 1, kernel_feat.shape[-1], device=kernel_feat.device, dtype=x.dtype)
        y_range = torch.linspace(-1, 1, kernel_feat.shape[-2], device=kernel_feat.device, dtype=x.dtype)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        
        kernel_feat = torch.cat([kernel_feat, coord_feat], 1)
        for i, kernel_layer in enumerate(self.kernel_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')    
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)
        
        # cate branch
        for i, cate_layer in enumerate(self.cate_convs):
            if i == self.cate_down_pos:
                seg_num_grid = self.seg_num_grids[idx]
                cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
            cate_feat = cate_layer(cate_feat)

        cate_pred = self.solo_cate(cate_feat)
        
        if eval:
           cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return kernel_pred, cate_pred

    @force_fp32(apply_to=('ins_preds', 'cate_preds'))
    def loss(self,
             ins_preds,
             cate_preds,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in
                         ins_preds]
        ins_label_list, cate_label_list, ins_ind_label_list = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            featmap_sizes=featmap_sizes)
        # ins
        ins_labels = [torch.cat([ins_labels_level_img[ins_ind_labels_level_img, ...]
                                 for ins_labels_level_img, ins_ind_labels_level_img in
                                 zip(ins_labels_level, ins_ind_labels_level)], 0)
                      for ins_labels_level, ins_ind_labels_level in zip(zip(*ins_label_list), zip(*ins_ind_label_list))]

        ins_preds = [torch.cat([ins_preds_level_img[ins_ind_labels_level_img, ...]
                                for ins_preds_level_img, ins_ind_labels_level_img in
                                zip(ins_preds_level, ins_ind_labels_level)], 0)
                     for ins_preds_level, ins_ind_labels_level in zip(ins_preds, zip(*ins_ind_label_list))]


        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.int().sum()
        
        # dice loss
        loss_ins = []
        for input, target in zip(ins_preds, ins_labels):
            if input.size()[0] == 0:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)
        return dict(
            loss_ins=loss_ins,
            loss_cate=loss_cate)

    def solo_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               featmap_sizes=None):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        for (lower_bound, upper_bound), stride, featmap_size, num_grid \
                in zip(self.scale_ranges, self.strides, featmap_sizes, self.seg_num_grids):

            ins_label = torch.zeros([num_grid ** 2, featmap_size[0], featmap_size[1]], dtype=torch.uint8, device=device)
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            output_stride = stride / 2

            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() < 10:
                   continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-self.limit_sample_num)
                down = min(down_box, coord_h+self.limit_sample_num)
                left = max(coord_w-self.limit_sample_num, left_box)
                right = min(right_box, coord_w+self.limit_sample_num)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                # ins
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list

    @force_fp32(apply_to=('seg_preds', 'cate_preds'))
    def get_seg(self, seg_preds, cate_preds, img_metas, cfg, rescale=None):
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = [
                seg_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    @force_fp32(apply_to=('segm_inputs'))
    def get_aug_seg(self, segm_inputs, img_metas_list, cfg, rescale=None):
        num_aug = len(segm_inputs)
        num_levels = len(segm_inputs[0][0])
        featmap_size = segm_inputs[0][0][0].size()[-2:]

        result_list = []
        for img_id in range(len(img_metas_list[0])):
            seg_pred_list = []
            cate_pred_list = []
            # for aug_id in range(num_aug):
            #     img_metas = img_metas_list[aug_id]
            #     seg_preds = segm_inputs[aug_id][0]
            #     cate_preds = segm_inputs[aug_id][1]
            #     cate_pred_list = [
            #         cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            #     ]
            #     seg_pred_list = [
            #         seg_preds[i][img_id].detach() for i in range(num_levels)
            #     ]

            #     cate_pred_list = torch.cat(cate_pred_list, dim=0)
            #     seg_pred_list = torch.cat(seg_pred_list, dim=0)
            #     seg_pred_list = F.interpolate(seg_pred_list.unsqueeze(1),
            #                       size=featmap_size,
            #                       mode='bilinear').squeeze(1)

            #     cate_pred_lists.append(cate_pred_list)
            #     seg_pred_lists.append(seg_pred_list)
            
            for i in range(num_levels):
                for aug_id in range(num_aug):
                    img_metas = img_metas_list[aug_id]
                    seg_preds = segm_inputs[aug_id][0]
                    cate_preds = segm_inputs[aug_id][1]
                    cate_level_pred = cate_preds[i][img_id].view(-1, self.cate_out_channels).detach()
                    seg_level_pred = seg_preds[i][img_id].detach()
                    seg_level_pred = F.interpolate(seg_level_pred.unsqueeze(1),
                                        size=featmap_size,
                                        mode='bilinear').squeeze(1)
                    seg_pred_list.append(seg_level_pred)
                    cate_pred_list.append(cate_level_pred)

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)
            img_shape = img_metas_list[0][img_id]['img_shape']
            scale_factor = img_metas_list[0][img_id]['scale_factor']
            ori_shape = img_metas_list[0][img_id]['ori_shape']

            result = self.get_seg_single(cate_pred_list, seg_pred_list,
                                            featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):
        assert len(cate_preds) == len(seg_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0) * 2
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores