# # Copyright (c) OpenMMLab. All rights reserved.
# import torch.nn as nn
# from mmcv.runner import Sequential
# from mmcv.runner import BaseModule
#
#
# class BidirectionalLSTM(nn.Module):
#
#     def __init__(self, nIn, nHidden, nOut):
#         super().__init__()
#
#         self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
#         self.embedding = nn.Linear(nHidden * 2, nOut)
#
#     def forward(self, input):
#         recurrent, _ = self.rnn(input)
#         T, b, h = recurrent.size()
#         t_rec = recurrent.view(T * b, h)
#
#         output = self.embedding(t_rec)  # [T * b, nOut]
#         output = output.view(T, b, -1)
#
#         return output
#
#
# class CRNNDecoder(BaseModule):
#     """Decoder for CRNN.
#
#     Args:
#         in_channels (int): Number of input channels.
#         num_classes (int): Number of output classes.
#         rnn_flag (bool): Use RNN or CNN as the decoder.
#         init_cfg (dict or list[dict], optional): Initialization configs.
#     """
#
#     def __init__(self,
#                  in_channels=None,
#                  num_classes=None,
#                  rnn_flag=False,
#                  init_cfg=dict(type='Xavier', layer='Conv2d'),
#                  **kwargs):
#         super().__init__(init_cfg=init_cfg)
#
#         self.num_classes = num_classes
#         self.rnn_flag = rnn_flag
#
#         if rnn_flag:
#             self.decoder = Sequential(
#                 BidirectionalLSTM(in_channels, 256, 256),
#                 BidirectionalLSTM(256, 256, num_classes))
#         else:
#             self.decoder = nn.Conv2d(
#                 in_channels, num_classes, kernel_size=1, stride=1)
#
#
#     def forward(self,
#                 feat,
#                 out_enc,
#                 targets_dict=None,
#                 img_metas=None,
#                 train_mode=True):
#         self.train_mode = train_mode
#         if train_mode:
#             return self.forward_train(feat, out_enc, targets_dict, img_metas)
#
#         return self.forward_test(feat, out_enc, img_metas)
#
#     def forward_train(self, feat, out_enc, targets_dict, img_metas):
#         """
#         Args:
#             feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.
#
#         Returns:
#             Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
#             :math:`C` is ``num_classes``.
#         """
#         assert feat.size(2) == 1, 'feature height must be 1'
#         if self.rnn_flag:
#             x = feat.squeeze(2)  # [N, C, W]
#             x = x.permute(2, 0, 1)  # [W, N, C]
#             x = self.decoder(x)  # [W, N, C]
#             outputs = x.permute(1, 0, 2).contiguous()
#         else:
#             x = self.decoder(feat)
#             x = x.permute(0, 3, 1, 2).contiguous()
#             n, w, c, h = x.size()
#             outputs = x.view(n, w, c * h)
#         return outputs
#
#     def forward_test(self, feat, out_enc, img_metas):
#         """
#         Args:
#             feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.
#
#         Returns:
#             Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
#             :math:`C` is ``num_classes``.
#         """
#         return self.forward_train(feat, out_enc, None, img_metas)
#
#
#