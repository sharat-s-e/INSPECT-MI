import yaml, argparse, time
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.visualize import scanlist_subgroup_check
from src.utils.data_loader import MRIDatasetFromDF
from src.utils.gpu_utils import get_least_used_gpu
from src.evaluate import evaluate
from src.models import build_model
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchinfo import summary

def main():
    parser = argparse.ArgumentParser(description='INSPECT-MI bias evaluation of model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    scanlist = Path(config_data['scanlist'])
    test_data_path = Path(config_data['test_data_path'])
    seg_data_path = Path(config_data['seg_data_path'])
    output_data_path = Path(config_data['output_data_path'])
    segmap = Path(config_data['segmap'])
    test_model = Path(config_data['test_model'])
    bm_model = Path(config_data.get('benchmark_model', None))
    subgroup_axis = config_data['subgroup_axis']
    diagnosis_dict = config_data['diagnosis_dict']
    batch_size = config_data['batch_size']
    model_name = config_data['model']
    device = get_least_used_gpu()
    print(f"Usage device: {device}")
    df = pd.read_csv(scanlist)
    scanlist_subgroup_check(df, subgroup_axis)
    df['data_path'] = df['data_path'].apply(lambda x: test_data_path / x)
    df['seg_path'] = df['seg_path'].apply(lambda x: seg_data_path / x)
    df['diagnosis'] = df['diagnosis'].apply(lambda x: diagnosis_dict[x])
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # Baseline evaluation
    test_ds = MRIDatasetFromDF(df)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    ip_data = next(iter(test_loader))
    ip_shape = np.shape(ip_data[0])
    model = build_model(model_name=model_name, num_classes=len(diagnosis_dict), in_channels=ip_shape[-4]).to(device)
    print(f"##### MODEL DETAILS #####")
    summary(model, ip_shape, device=device)
    print(f"#########################\n")
    print(f"[{time.strftime('%X')}] Loading model weights: {test_model}")
    model.load_state_dict(torch.load(test_model, map_location=device))
    test_loss, test_acc, y_true, y_pred, y_probs, y_logits = evaluate(model, test_loader, device, criterion)
    print(test_loss, test_acc)
    
if __name__ == "__main__":
    main()