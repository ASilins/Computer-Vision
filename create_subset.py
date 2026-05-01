import pickle
import os

# Get the absolute path to your current directory
base_path = os.getcwd()

def create_subset(filename, ratio=10):
    input_path = os.path.join(base_path, 'data/nuScenes', filename)
    output_path = input_path.replace('.pkl', '_subset_10.pkl')
    
    if not os.path.exists(input_path):
        print(f"ERROR: Cannot find {input_path}")
        return

    print(f"Processing {filename}...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Slice the data
    if isinstance(data, dict) and 'infos' in data:
        data['infos'] = data['infos'][::ratio]
    else:
        data = data[::ratio]
        
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"SUCCESS: Saved to {output_path}")

create_subset('infos_train_10sweeps_withvelo_filter_True.pkl')
create_subset('infos_val_10sweeps_withvelo_filter_True.pkl')
