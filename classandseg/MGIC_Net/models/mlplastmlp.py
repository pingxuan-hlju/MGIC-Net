import torch
from initialize import *
from models.layers.grid_attention_layer import GridAttentionBlock3D
from models.networks_other_3D import init_weights
from models.utils_3D import UnetConv3, conv3DBatchNorm, conv3DBatchNormRelu
import torch.nn.functional as F


class mlplastmlp(nn.Module):
    def __init__(self, num_classes, attention=True, in_channels=3, feature_scale=4, is_batchnorm=True,
                 aggregation_mode='concat', nonlocal_mode='concatenation'):
        super().__init__()
        self.attention = attention
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.aggregation_mode = aggregation_mode
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]  # ÿ��block�����ͨ������
        attention_dsample = (2, 2, 2)

        ################
        # Feature Extraction
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv5 = UnetConv3(filters[3], filters[3], self.is_batchnorm)

        # adaptation layer
        self.conv5_p = conv3DBatchNormRelu(filters[3], filters[2], 1, 1, 0)
        self.conv6_p = conv3DBatchNorm(filters[2], num_classes, 1, 1, 0)

        # text encoder
        self.oh_linear = torch.nn.Sequential(torch.nn.Linear(83,32))
        self.oh_conv1d = torch.nn.Conv1d(83, 30, kernel_size=1, stride=1)
        # self.maxpool1d = torch.nn.MaxPool1d(kernel_size=1)
        self.avg_pool1d = torch.nn.AvgPool1d(kernel_size=1)
        self.oh_batch_norm = torch.nn.BatchNorm1d(30)
        self.num_linear = torch.nn.Sequential(torch.nn.Linear(57,32))
        self.num_conv1d = torch.nn.Conv1d(57, 30, kernel_size=1, stride=1)
        self.num_batch_norm = torch.nn.BatchNorm1d(30)

        self.encoder_relu = torch.nn.ReLU()

        # CMSA
        sub = 2
        # out_dim = [218, 80, 128]
        out_dim = [192, 80, 128]
        self.conv_cmsa1 = torch.nn.Conv3d(in_channels=out_dim[1], out_channels=int(out_dim[1] / 2), kernel_size=1)
        self.conv_cmsa2 = torch.nn.Conv3d(in_channels=out_dim[0], out_channels=int(out_dim[0] / 2), kernel_size=1)
        self.conv_cmsa3 = torch.nn.Conv3d(in_channels=out_dim[0], out_channels=out_dim[0], kernel_size=1)

        self.maxpool_cmsa = torch.nn.MaxPool3d(kernel_size=sub, stride=sub, padding=[1, 0, 0])  # 1,0,0

        self.final_conv_cmsa1 = torch.nn.Conv3d(in_channels=int(out_dim[1] / 2), out_channels=out_dim[1], kernel_size=1)
        self.final_conv_cmsa2 = torch.nn.Conv3d(in_channels=int(out_dim[0] / 2), out_channels=out_dim[0], kernel_size=1)
        self.final_conv_cmsa3 = torch.nn.Conv3d(in_channels=out_dim[0], out_channels=out_dim[0], kernel_size=1)

        self.relu_cmsa = torch.nn.ReLU()
        self.softmax_cmsa1 = torch.nn.Softmax(dim=1)
        self.softmax_cmsa = torch.nn.Softmax()
        self.softmax_cmsa2 = torch.nn.Softmax(dim=-1)

        self.avgpool3d_cmsa = torch.nn.AdaptiveAvgPool3d([3, 6, 6])

        ####################
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

        self.classifier = nn.Sequential(nn.Linear(32 + 32 + 192, 128),nn.ReLU(),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,num_classes))
    def aggregation_concat(self, *attended_maps):  # ok

        return self.classifier(torch.cat(attended_maps, dim=1))  # [10,2]

    def CMSA_layer(self, in_feats, dim, conv, maxpool, final_conv, relu_cmsa, softmax_cmsa):

        # query
        query = conv(in_feats)  # [10,161,1,3,3]
        query = relu_cmsa(query)  # [10, 64, 3, 6, 6]
        query = torch.reshape(query, [in_feats.shape[0], dim, -1])  # [10, 161, 9]
        query = torch.transpose(query, 2, 1)  # [10, 9, 161] instead of transposing key, i transposed query
        # print("after transpose query:", query.shape)

        # key
        key = conv(in_feats)  # [10, 161, 1, 3, 3]
        #   print("key after conv:", key.shape)
        key = relu_cmsa(key)

        # key = maxpool(key)  # [10, 64, 1, 1, 1]

        #   print("key after maxpool: ", key.shape)

        key = torch.reshape(key, [in_feats.shape[0], dim, -1])  # [10, 161, 1]
        #    print("key after reshape", key.shape)

        # key = torch.transpose(key,2,1)
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
        value = conv(in_feats)  # [10, 64, 1, 1, 1]
        # print("value", value.shape)
        value = relu_cmsa(value)
        # value = maxpool(value)  #
        # print("value maxpool", value.shape)

        value = torch.reshape(value, [in_feats.shape[0], dim, -1])  # [10, 161, 1]
        # print("value after reshape", value.shape)

        value = torch.transpose(value, 2, 1)  # [10, 18, 64]
        # print("value after transpose", value.shape)

        # VALUE X (QUERY X KEY)
        final_fea = torch.matmul(queryXkey, value)  # [10, 108, 64]
        # print("final shape", final_fea.shape)
        final_fea = torch.transpose(final_fea, 2, 1)  # [10,161,9]
        final_fea = torch.reshape(final_fea, [in_feats.shape[0], dim, -1, in_feats.shape[3],
                                              in_feats.shape[4]])  # [10, 64, 3, 6, 6]
        # print("final 5 dimension tensor", final_fea.shape)

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

        # t = torch.reshape(t, [t.shape[0], attr, 1])  # [b,71,1]

        t = linear_encoder(t.float())
        t = t.reshape(-1,out_attr,1)
        # t = batch_norm(t)

        # t = self.encoder_relu(t)
        # silu = nn.SiLU()
        # t = silu(t)

        # t = self.avg_pool1d(t)

        t = t.repeat(1, 1, np.prod(in_shape))  # [b,30,9]

        t = torch.reshape(t, [t.shape[0], out_attr, in_shape[0], in_shape[1], in_shape[2]])  # [b,30, 1, 3, 3]

        return t

    def forward(self, x, oneHot, num):  # x:(batch) * 3(views) * 3 rgb channel*27(depth) *im_size* im_size
        x = x.transpose(0, 1)

        view_pool = []
        view_cmsa = []
        view_g1 = []
        view_g2 = []

        for v in x:  # 3 views
            # Feature Extraction
            conv1 = self.conv1(v)
            maxpool1 = self.maxpool1(conv1)

            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)

            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)

            conv4 = self.conv4(maxpool3)
            maxpool4 = self.maxpool4(conv4)

            conv5 = self.conv5(maxpool4)  # torch.Size([1, 128, 10, 10, 10])
            t_onehot = self.catnum_tensor(oneHot, self.oh_linear, self.oh_batch_norm, 83, 32,
                                          [10, 3, 3])  # shape:torch.Size([2, 30, 10, 10, 10])
            t_num = self.catnum_tensor(num, self.num_linear, self.num_batch_norm, 57, 32, [10, 3, 3]) #torch.Size([2, 60, 10, 10, 10])

            in_feat5 = torch.cat([conv5, t_onehot, t_num], 1)  #torch.Size([2, 218, 10, 10, 10])
            in_feat5 = in_feat5.type(torch.cuda.FloatTensor)  # .to('cuda')
            cmsa_gconv3 = self.CMSA_layer(in_feat5, dim=192, conv=self.conv_cmsa3, maxpool=self.maxpool_cmsa,
                                          final_conv=self.final_conv_cmsa3, relu_cmsa=self.relu_cmsa,
                                          softmax_cmsa=self.softmax_cmsa)

            # conv5_p = self.conv5_p(conv5)
            # conv6_p = self.conv6_p(conv5_p)

            batch_size = v.shape[0]

            # mvcnn style
            pooled = torch.sum(conv5.view(batch_size, conv5.shape[1], -1), dim=-1)
            g1 = torch.sum(t_onehot.view(batch_size, t_onehot.shape[1], -1), dim=-1)  # [1,143]
            g2 = torch.sum(t_num.view(batch_size, t_num.shape[1], -1), dim=-1)
            pooled_cmsa = torch.sum(cmsa_gconv3.view(batch_size, cmsa_gconv3.shape[1], -1), dim=-1)
            # pooled = F.adaptive_avg_pool3d(conv6_p, (1, 1,1)).view(batch_size, -1)  # sonnet style

            view_pool.append(pooled)
            view_g1.append(g1)
            view_g2.append(g2)
            view_cmsa.append(pooled_cmsa)

        # mvcnn style
        pooled_view = view_pool[0]
        g1_view = view_g1[0]  # [10,64]
        g2_view = view_g2[0]
        pooled_cmsa_view = view_cmsa[0]

        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])
            g1_view = torch.max(g1_view, view_g1[i])
            g2_view = torch.max(g2_view, view_g2[i])
            pooled_cmsa_view = torch.max(pooled_cmsa_view, view_cmsa[i])

        pooled_view = self.aggregation_concat(g1_view, g2_view, pooled_cmsa_view) 
        #g1_view:torch.Size([2, 30]) g1_view2:torch.Size([2, 60]) pooled_cmsa_view:torch.Size([2, 218])

        return pooled_view
