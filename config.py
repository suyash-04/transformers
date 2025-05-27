from pathlib import Path
import regex as re 
def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 500,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "nepali",
        "lang_tgt": "english",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }
    

def get_weights_file_path(config , epoch : str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename   = f"{model_basename}{epoch}.pt"
    return str(Path('.')/ model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    def extract_epoch(file_path):
        match = re.search(r"(\d+)", file_path.stem)
        return int(match.group(1)) if match else -1

    weights_files.sort(key=extract_epoch)
    return str(weights_files[-1])
    
    