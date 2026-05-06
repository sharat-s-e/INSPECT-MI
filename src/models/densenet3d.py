from monai.networks.nets import DenseNet121

def densenet121_3d(num_classes=2, in_channels=1, dropout=0.3):
    return DenseNet121(
        spatial_dims=3,   # 3D images (MRI slices)
        in_channels=in_channels,    # Single channel input (grayscale MRI images)
        out_channels=num_classes,    # Number of classes for classification (modify as needed)
        dropout_prob=dropout
    )   