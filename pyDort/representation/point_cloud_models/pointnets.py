import torch
import torch.nn as nn
import torch.nn.functional as F


# from .pooling import Pooling
class Pooling(torch.nn.Module):
    def __init__(self, pool_type='max'):
        self.pool_type = pool_type
        super(Pooling, self).__init__()

    def forward(self, input):
        if self.pool_type == 'max':
            return torch.max(input, 2)[0].contiguous()
        elif self.pool_type == 'avg' or self.pool_type == 'average':
            return torch.mean(input, 2).contiguous()

class PointNet(torch.nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bcn", use_bn=True, global_feat=True):
        # emb_dims:			Embedding Dimensions for PointNet.
        # input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
        super(PointNet, self).__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
        self.input_shape = input_shape
        self.emb_dims = emb_dims
        self.use_bn = use_bn
        self.global_feat = global_feat
        if not self.global_feat: self.pooling = Pooling('max')

        self.layers = self.create_structure()

    def create_structure(self):
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, self.emb_dims, 1)
        self.relu = torch.nn.ReLU()

        if self.use_bn:
            self.bn1 = torch.nn.BatchNorm1d(64)
            self.bn2 = torch.nn.BatchNorm1d(64)
            self.bn3 = torch.nn.BatchNorm1d(64)
            self.bn4 = torch.nn.BatchNorm1d(128)
            self.bn5 = torch.nn.BatchNorm1d(self.emb_dims)

        if self.use_bn:
            layers = [self.conv1, self.bn1, self.relu,
                        self.conv2, self.bn2, self.relu,
                        self.conv3, self.bn3, self.relu,
                        self.conv4, self.bn4, self.relu,
                        self.conv5, self.bn5, self.relu]
        else:
            layers = [self.conv1, self.relu,
                        self.conv2, self.relu,
                        self.conv3, self.relu,
                        self.conv4, self.relu,
                        self.conv5, self.relu]
        return layers


    def forward(self, input_data):
        # input_data: 		Point Cloud having shape input_shape.
        # output:			PointNet features (Batch x emb_dims)
        if self.input_shape == "bnc":
            num_points = input_data.shape[1]
            input_data = input_data.permute(0, 2, 1)
        else:
            num_points = input_data.shape[2]
        if input_data.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        output = input_data
        for idx, layer in enumerate(self.layers):
            output = layer(output)
            if idx == 1 and not self.global_feat: point_feature = output

        if self.global_feat:
            return output
        else:
            output = self.pooling(output)
            output = output.view(-1, self.emb_dims, 1).repeat(1, 1, num_points)
            return torch.cat([output, point_feature], 1)


class PointNetClassifier(nn.Module):
    def __init__(self, feature_model=PointNet(), num_classes=40):
        super(PointNetClassifier, self).__init__()
        self.feature_model = feature_model
        self.num_classes = num_classes

        self.linear1 = torch.nn.Linear(self.feature_model.emb_dims, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.dropout1 = torch.nn.Dropout(p=0.7)
        self.linear2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.dropout2 = torch.nn.Dropout(p=0.7)
        self.linear3 = torch.nn.Linear(256, self.num_classes)

        self.pooling = Pooling('max')

    def forward(self, input_data):
        output = self.pooling(self.feature_model(input_data))
        # output = F.relu(self.bn1(self.linear1(output)))
        # output = self.dropout1(output)
        # output = F.relu(self.bn2(self.linear2(output)))
        # output = self.dropout2(output)
        # output = self.linear3(output)
        return output
