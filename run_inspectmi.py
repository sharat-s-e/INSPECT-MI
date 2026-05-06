import yaml, argparse
from pathlib import Path
import pandas as pd
from src.utils.visualize import scanlist_subgroup_check

def main():
    parser = argparse.ArgumentParser(description='INSPECT-MI bias evaluation of model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    scanlist = Path(config_data['scanlist'])
    diagnosis_col = config_data['diagnosis_col']
    test_data_path = Path(config_data['test_data_path'])
    seg_data_path = Path(config_data['seg_data_path'])
    output_data_path = Path(config_data['output_data_path'])
    segmap = Path(config_data['segmap'])
    test_model = Path(config_data['test_model'])
    bm_model = Path(config_data.get('benchmark_model', None))
    subgroup_axis = config_data['subgroup_axis']
    df = pd.read_csv(scanlist)
    scanlist_subgroup_check(df, subgroup_axis, diagnosis_col)
    
if __name__ == "__main__":
    main()