#%%
import yaml

#%%
def load_config(config_path):
    """
    載入並回傳 YAML 設定檔的內容
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


#%%
if __name__ == "__main__":

    File = "../../config/data_config.yaml"
    config = load_config(File)
    print(config)

    for key, value in config.items():
        print(f"{key}: {value}")


# %%
