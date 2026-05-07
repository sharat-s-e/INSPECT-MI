import yaml, argparse, time, os
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.visualize import scanlist_subgroup_check, plot_confusion_matrix, plot_roc_curve, get_eval_metrics, group_analysis, plot_hbar
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
    scanlist = Path(config_data['path_params']['scanlist_path'])
    test_data_path = Path(config_data['path_params']['test_data_path'])
    seg_data_path = Path(config_data['path_params']['seg_data_path'])
    output_dir = Path(config_data['path_params']['output_dir'])
    atlas_map = Path(config_data['atlas_params']['atlas_map'])
    atlas_locs = config_data['atlas_params']['atlas_locs']
    model_name = config_data['model_params']['model']
    test_model = Path(config_data['model_params']['test_model'])
    subgroup_axis = config_data['eval_params']['subgroup_axis']
    diag_dict = config_data['eval_params']['diagnosis_dict']
    idx_dict = {v: k for k, v in diag_dict.items()}
    batch_size = config_data['eval_params']['batch_size']
    device = get_least_used_gpu()
    print(f"Usage device: {device}")
    df = pd.read_csv(scanlist)
    scanlist_subgroup_check(df, subgroup_axis)
    tdf = df.copy()
    tdf['data_path'] = tdf['data_path'].apply(lambda x: test_data_path / x)
    tdf['seg_path'] = tdf['seg_path'].apply(lambda x: seg_data_path / x)
    tdf['diagnosis'] = tdf['diagnosis'].apply(lambda x: diag_dict[x])
    hc_mask = (df['diagnosis']==0) #Label '0' is assigned to healthy controls
    criterion = nn.CrossEntropyLoss(reduction='sum')

    #Testing workflow
    test_ds = MRIDatasetFromDF(tdf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    ip_data = next(iter(test_loader))
    ip_shape = np.shape(ip_data[0])
    print(f"[{time.strftime('%X')}] Building model")
    model = build_model(model_name=model_name, num_classes=len(diag_dict), in_channels=ip_shape[-4]).to(device)
    print(f"##### MODEL DETAILS #####")
    summary(model, ip_shape, device=device)
    print(f"#########################")
    print(f"[{time.strftime('%X')}] Loading model weights: {test_model}")
    model.load_state_dict(torch.load(test_model, map_location=device))
    print(f"[{time.strftime('%X')}] Model building completed")
    print(f"\n====== Evaluation results ======")
    test_loss, test_acc, y_true, y_pred, y_probs, y_logits = evaluate(model, test_loader, device, criterion)
    print(f"[{time.strftime('%X')}] Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    plot_confusion_matrix(y_true, y_pred, class_names=list(diag_dict.keys()), save_path=os.path.join(output_dir, f'test_confusion_matrix_{scanlist.stem}.png'))
    plot_roc_curve(y_true, y_probs, save_path=os.path.join(output_dir, f'test_roc-curve_{scanlist.stem}.png'))
    cm, precision, recall = get_eval_metrics(y_true, y_pred, pos_label=1, average='binary')
    print(f"[{time.strftime('%X')}] Confusion Matrix:\n{cm}")
    print(f"[{time.strftime('%X')}] Precision: {precision:.4f}, Recall: {recall:.4f}")
    for idx, (true_label, pred_label, pred_prob, pred_logits) in enumerate(zip(y_true, y_pred, y_probs, y_logits)):
        df.at[idx, 'prediction'] = idx_dict[pred_label]
        df.at[idx, 'class_probability'] = pred_prob
        df.at[idx, 'matching'] = 1 if pred_label == true_label else 0
        for k,v in diag_dict.items():
            df.at[idx, f'{k}_logit'] = pred_logits[v]
        df.to_csv(scanlist, index=False)
    print(f"================================")

    #Bias computation
    print(f"\n======== Bias analysis =========")
    grp_metrics = {}
    for ax in subgroup_axis:
        grp_metrics[ax] = group_analysis(
            csv_file=scanlist,
            label_col='diagnosis',
            predict_col='prediction',
            prob_col='class_probability',
            sensitive_col=ax,
            positive_label=idx_dict[1],
            output_file = os.path.join(output_dir, f'test_bias-analysis_{scanlist.stem}.txt')
        )
    print(f"================================")

    #TSS calcuation
    print(f"\n======= TSS computation ========")
    mapping_dict = pd.read_csv(atlas_map).set_index('Key')['Location'].to_dict()
    # Baseline evaluation
    print(f"[{time.strftime('%X')}] Starting baseline evaluation")
    test_ds = MRIDatasetFromDF(tdf[~hc_mask])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    _, _, _, _, base_probs, _ = evaluate(model, test_loader, device, criterion)
    print(f"[{time.strftime('%X')}] Completed baseline evaluation")
    # Atlas masked evaluation
    print(f"[{time.strftime('%X')}] Starting saliency evaluation")
    result_summary = {}
    for aidx, atlas_index in enumerate(atlas_locs):
        print(f"[{time.strftime('%X')}] Starting evaluation with atlas index: {atlas_index}")
        test_ds = MRIDatasetFromDF(tdf[~hc_mask], atlas_loc=atlas_index)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        _, _, _, _, probs, _ = evaluate(model, test_loader, device, criterion)
        sscore = np.mean(np.array(base_probs)-np.array(probs))
        result_summary[atlas_index] = sscore
        print(f"[{time.strftime('%X')}] Saliency score for {mapping_dict[atlas_index]}: {sscore}")
        print(f"[{time.strftime('%X')}] Completed evaluation with atlas index: {atlas_index}")
    mean_base_probs = np.mean(np.array(base_probs))
    tss_val = np.sum(np.array([result_summary[aloc] for aloc in atlas_locs]))
    summary_df = pd.DataFrame.from_dict(result_summary, orient='index', columns=['Probs difference'])
    summary_df.index.name = 'Atlas Index'
    summary_df = summary_df.sort_index(key=lambda x: x.astype(int))
    summary_df['Atlas Location'] = summary_df.index.map(mapping_dict)
    summary_df = summary_df.reset_index()
    plot_hbar(summary_df, ['Atlas Location','Probs difference'], os.path.join(output_dir,f'Atlas_Masking_Summary.png'))
    base_row = pd.DataFrame({'Atlas Index': 'Baseline', 'Atlas Location': 'Baseline', 'Probs difference': mean_base_probs}, index=[0])
    tss_row = pd.DataFrame({'Atlas Index': 'TSS', 'Atlas Location': 'Sum', 'Probs difference': tss_val}, index=[0])
    summary_df = pd.concat([base_row, summary_df, tss_row], ignore_index=True)
    summary_df.to_csv(os.path.join(output_dir,f'Atlas_Masking_Summary.csv'), columns=['Atlas Index','Atlas Location','Probs difference'], index=False, float_format='%.4f')
    print(f"[{time.strftime('%X')}] TSS Value: {tss_val}")
    print(f"[{time.strftime('%X')}] Completed saliency evaluation")
    print(f"================================")

    #Summary
    print(f"\n=========== Summary ============")
    report = ""
    for ax in subgroup_axis:
        met_dict = grp_metrics[ax]['f1score']
        grp_vals = [gval for gval in met_dict.keys() if (gval not in ['overall','bias'])]
        met_vals = np.array([met_dict[gval] for gval in grp_vals])
        report += f"Evaluation axis: {ax} - {grp_vals}\n"
        report += f"{'Worse performing group':30s}: {grp_vals[np.argmin(met_vals)]}\n"
        report += f"{'Worse F1 score':30s}: {np.min(met_vals):.4f}\n"
        report += f"{'Bias value':30s}: {met_dict['bias']:.4f}\n"
        report += "\n"
    report += f"{'TSS Value':30s}: {tss_val:.4f}"
    print(report)
    with open(os.path.join(output_dir,f'Summary.txt'),'a') as f:
        f.write(report + "\n\n")
    print(f"================================")

if __name__ == "__main__":
    main()