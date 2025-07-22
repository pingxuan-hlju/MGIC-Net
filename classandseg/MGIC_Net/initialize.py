import torch.nn as nn
import numpy as np
"""
初始化神经网络模块的权重。每个函数都接受一个模块作为参数，并遍历该模块的所有子模块。
对于每个子模块，函数检查它是否是 nn.Conv2d、nn.BatchNorm2d 或 nn.Linear 类型的实例。
如果是，则使用相应的初始化方法初始化其权重和偏差。
# """
# def weights_init_kaimingUniform(module):
#     for m in module.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.uniform_(m.weight, a=0, b=1)
#             nn.init.constant_(m.bias, val=0.)
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, val=0.)

# def weights_init_kaimingNormal(module):
#     for m in module.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.normal_(m.weight, 0, 0.01)
#             nn.init.constant_(m.bias, val=0.)
#         elif isinstance(m, nn.Linear):
#             nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, val=0.)

# def weights_init_xavierUniform(module):
#     for m in module.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.uniform_(m.weight, a=0, b=1)
#             nn.init.constant_(m.bias, val=0.)
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, val=0.)

# def weights_init_xavierNormal(module):
#     for m in module.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.normal_(m.weight, 0, 0.01)
#             nn.init.constant_(m.bias, val=0.)
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, val=0.)
