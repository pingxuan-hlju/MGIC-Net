import torch
from nnunet.initialize import *
from nnunet.models.layers.grid_attention_layer import GridAttentionBlock3D
from nnunet.models.utils_3D import UnetConv3, conv3DBatchNorm, conv3DBatchNormRelu
import torch.nn.functional as F
from nnunet.utilities.nd_softmax import softmax_helper

import torch
import torch.nn as nn
from torch.nn import Dropout,LayerNorm,Linear,Softmax
import math
import copy
import yaml
class MV3Dunet_down_text_cmsa_mid(nn.Module):
    def __init__(self, num_classes, attention=True, in_channels=3, feature_scale=4, is_batchnorm=True,
                 aggregation_mode='concat', nonlocal_mode='concatenation'):
        super(MV3Dunet_down_text_cmsa_mid, self).__init__()
        self.attention = attention
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.aggregation_mode = aggregation_mode
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale)
                   for x in filters]  # 每个block输出的通道数量
        attention_dsample = (2, 2, 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
        ################
        # Feature Extraction
        # self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        # self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv1 = UnetConv3(320, 128, self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.onehot_emb = nn.Linear(1, 128)
        self.num_emb = nn.Linear(1, 128)
        self.q_mlp = nn.Linear(128, 128)

        # text encoder
        self.final_mlp = nn.Linear(128, 128)
        self.oh_conv1d = torch.nn.Conv1d(57, 30, kernel_size=1, stride=1)
        # self.maxpool1d = torch.nn.MaxPool1d(kernel_size=1)
        self.avg_pool1d = torch.nn.AvgPool1d(kernel_size=1)
        self.oh_batch_norm = torch.nn.BatchNorm1d(30)

        self.num_conv1d = torch.nn.Conv1d(57, 30, kernel_size=1, stride=1)
        self.num_batch_norm = torch.nn.BatchNorm1d(30)

        self.encoder_relu = torch.nn.ReLU()
        self.emb_1 = nn.Linear(128,128)
        self.emb_2 = nn.Linear(128,128)
        self.final_1 = nn.Linear(128,128)
        self.final_2 = nn.Linear(128,128)
        self.final_3 = nn.Linear(128,128)
        self.final_4 = nn.Linear(128,128)
        self.gcn_mlp1 = nn.Linear(128,128)
        self.gcn_mlp2 = nn.Linear(128,128)
        self.norm1 = nn.LayerNorm(128, eps=1e-6)
        self.norm2 = nn.LayerNorm(128, eps=1e-6)
        # CMSA
        sub = 2
        # out_dim = [218, 80, 128]
        out_dim = [188, 80, 128]
        self.conv_cmsa1 = torch.nn.Conv3d(
            in_channels=out_dim[1], out_channels=int(out_dim[1] / 2), kernel_size=1)
        self.conv_cmsa2 = torch.nn.Conv3d(
            in_channels=out_dim[0], out_channels=int(out_dim[0] / 2), kernel_size=1)
        self.conv_cmsa3 = torch.nn.Conv3d(
            in_channels=out_dim[0], out_channels=out_dim[0], kernel_size=1)

        self.maxpool_cmsa = torch.nn.MaxPool3d(
            kernel_size=sub, stride=sub, padding=[1, 0, 0])  # 1,0,0

        self.final_conv_cmsa1 = torch.nn.Conv3d(in_channels=int(
            out_dim[1] / 2), out_channels=out_dim[1], kernel_size=1)
        self.final_conv_cmsa2 = torch.nn.Conv3d(in_channels=int(
            out_dim[0] / 2), out_channels=out_dim[0], kernel_size=1)
        self.final_conv_cmsa3 = torch.nn.Conv3d(
            in_channels=out_dim[0], out_channels=out_dim[0], kernel_size=1)

        self.relu_cmsa = torch.nn.ReLU()
        self.softmax_cmsa1 = torch.nn.Softmax(dim=1)
        self.softmax_cmsa = torch.nn.Softmax()
        self.softmax_cmsa2 = torch.nn.Softmax(dim=-1)

        self.avgpool3d_cmsa = torch.nn.AdaptiveAvgPool3d([3, 6, 6])
        self.classmlp = nn.Linear(128,num_classes)
        ####################
        # initialise weights
        self.weight = nn.Parameter(torch.rand(1))

        self.classifier = nn.Linear(57 + 57 + 128, num_classes)

    def aggregation_concat(self, *attended_maps):  # ok

        return self.classifier(torch.cat(attended_maps, dim=1))  # [10,2]

    def crossatt(self, img, text, img_emb, text_emb, img_final, text_final,it_final,ti_final, relu_cmsa):
        img_q = img_emb(img)
        img_q = relu_cmsa(img_q)
        img_k = img_emb(img)
        img_k = relu_cmsa(img_k)
        img_k = torch.transpose(img_k, 2, 1)
        img_v = img_emb(img)
        img_v = relu_cmsa(img_v)
        text_q = text_emb(text)
        text_q = relu_cmsa(text_q)
        text_k = text_emb(text)
        text_k = relu_cmsa(text_k)
        text_k = torch.transpose(text_k, 2, 1)
        text_v = text_emb(text)
        text_v = relu_cmsa(text_v)
        imgqxk = torch.matmul(img_q, img_k)
        imgqxk=self.softmax_cmsa2(imgqxk)
        img_fea = torch.matmul(imgqxk, img_v)
        textqxk = torch.matmul(text_q, text_k)
        textqxk=self.softmax_cmsa2(textqxk)
        text_fea = torch.matmul(textqxk, text_v)

        itqxk = torch.matmul(img_q, text_k)
        itqxk=self.softmax_cmsa2(itqxk)
        it_fea = torch.matmul(itqxk, text_v)

        tiqxk = torch.matmul(text_q, img_k)
        tiqxk=self.softmax_cmsa2(tiqxk)
        ti_fea = torch.matmul(tiqxk, img_v)

        img_fea = img_final(img_fea)
        text_fea = text_final(text_fea)
        it_fea = it_final(it_fea)
        ti_fea = ti_final(ti_fea)

        return img_fea+it_fea+img ,text_fea+ti_fea+text

    def gcn(self, tensor, mlp, relu):
        # if tensor.requires_grad:
        #     tensor = tensor.half()
        tensor = tensor.half()
        # 计算绝对值的差并计算指数
        distances = torch.exp(-torch.sum(torch.abs(tensor.unsqueeze(2) - tensor.unsqueeze(1)), dim=3)).detach()
        tensor = tensor.float()
        distances = distances.float()
        output = relu(mlp(torch.matmul(distances, tensor)))
        tensor = tensor.float()
        return output + tensor

    def CMSA_layer(self, in_feats, dim, q_emb, maxpool, final_conv, relu_cmsa, softmax_cmsa):

        # query
        query = q_emb(in_feats)  # [10,161,1,3,3]
        query = relu_cmsa(query)  # [10, 64, 3, 6, 6]

        # print("after transpose query:", query.shape)

        # key
        key = q_emb(in_feats)  # [10, 161, 1, 3, 3]
        #   print("key after conv:", key.shape)
        key = relu_cmsa(key)

        key = torch.transpose(key, 2, 1)
        # print("after transpose", key.shape)

        # first multiplication: query X key
        # QUERY X KEY -> SOFTMAX
        queryXkey = torch.matmul(query, key)  # [10, 9, 1]
        # print("query x key = ", queryXkey.shape)
        # queryXkey1 = self.softmax_cmsa1(queryXkey)  # [10, 9, 1]
        # queryXkey2 = self.softmax_cmsa2(queryXkey)
        queryXkey = self.softmax_cmsa2(queryXkey)
        # print("after softmax:", queryXkey.shape)

        # value
        value = q_emb(in_feats)  # [10, 64, 1, 1, 1]
        # print("value", value.shape)
        value = relu_cmsa(value)
        # value = maxpool(value)  #
        # print("value maxpool", value.shape)
        # VALUE X (QUERY X KEY)
        final_fea = torch.matmul(queryXkey, value)  # [10, 108, 64]
        # print("final shape", final_fea.shape)

        # last conv layer
        final_fea = final_conv(final_fea)  # [10, 128, 3, 6, 6]
        final_fea = relu_cmsa(final_fea)
        final_fea = final_fea + in_feats  # [10, 128, 3, 6, 6]
        # print(final_fea.shape)

        # avg pooling
        # final_fea = self.avgpool3d_cmsa(final_fea)
        # final_fea = torch.mean(final_fea, dim=2)  # [10, 128, 6, 6]
        # print("after avg pooling", final_fea.shape)

        return final_fea

    def catnum_tensor(self, t, linear_encoder, batch_norm, attr, out_attr, in_shape):
        """input:oneHot, self.oh_conv1d, self.oh_batch_norm, 71, 30, [1, 3, 3]
           num, self.num_conv1d, self.num_batch_norm, 7, 3, [1, 3, 3]"""

        # self.nn_out = torch.nn.Linear(attr, int(attr/2))
        # t = linear_encoder(t.float())

        t = torch.reshape(t, [t.shape[0], attr, 1])  # [b,71,1]

        t = linear_encoder(t.float())

        t = batch_norm(t)  # 显存不够

        # t = self.encoder_relu(t)
        silu = nn.SiLU()
        t = silu(t)

        # t = self.avg_pool1d(t)

        t = t.repeat(1, 1, np.prod(in_shape))  # [b,30,9]

        t = torch.reshape(t, [t.shape[0], out_attr, in_shape[0],
                          in_shape[1], in_shape[2]])  # [b,30, 1, 3, 3]

        return t

    # x:(batch) * 3(views) * 3 rgb channel*27(depth) *im_size* im_size
    def forward(self, x, oneHot, num):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1) 

        conv1 = self.conv1(x)
        onehot = oneHot.unsqueeze(-1)
        num = num.unsqueeze(-1)
        onehot = self.onehot_emb(onehot)
        num = self.num_emb(num)
        conv1 = conv1.flatten(2).transpose(-1, -2)
        conv1 = torch.cat((cls_tokens, conv1), dim=1)
        # torch.Size([2, 218, 10, 10, 10])
        text = torch.cat([onehot, num], 1)
        conv1 = self.gcn(conv1,self.gcn_mlp1,self.relu_cmsa)
        text = self.gcn(text,self.gcn_mlp2,self.relu_cmsa)
        conv1 = 0.1 * conv1
        text = 0.9 * text
        cross_imgfea,cross_textfea = self.crossatt(self.norm1(conv1),self.norm2(text),self.emb_1,self.emb_2,self.final_1,self.final_2,self.final_3,self.final_4,relu_cmsa=self.relu_cmsa)  # 交叉注意力
        cross_imgfea = cross_imgfea + conv1
        cross_textfea = cross_textfea + text
        in_feat5 = torch.cat([cross_imgfea, cross_textfea], 1)
        in_feat5 = in_feat5.type(torch.cuda.FloatTensor)  # .to('cuda')
        cmsa_gconv3 = in_feat5 + self.CMSA_layer(self.norm1(in_feat5), dim=188, q_emb=self.q_mlp, maxpool=self.maxpool_cmsa,
                                      final_conv=self.final_mlp, relu_cmsa=self.relu_cmsa,
                                      softmax_cmsa=self.softmax_cmsa)
        batch_size = x.shape[0]

        # mvcnn style
        # g1 = torch.sum(onehot, dim=-1)  # [1,143]
        # g2 = torch.sum(num, dim=-1)
        # pooled_cmsa = torch.sum(cmsa_gconv3, dim=1)

        # pooled_view = self.aggregation_concat(g1, g2, pooled_cmsa)
        output = self.classmlp(cmsa_gconv3[:,0])
        # output = torch.sigmoid(output)
        return output
