import yaml
from datetime import datetime
from yaml.loader import SafeLoader


def yaml_processor(config_file_path, is_debug):
    with open(config_file_path, 'r') as f:
        data = yaml.load(f, Loader=SafeLoader)

    daytime = datetime.now()
    date = daytime.strftime("%y_%m%d_%H%M")
    model_name = data["Model"]["name"]
    train_root_path = data["Training"]["root_path"]
    test_root_path = data["Testing"]["root_path"]
    train_root_path = train_root_path.replace(r"${Model.name}", model_name)
    test_root_path = test_root_path.replace(r"${Model.name}", model_name)

    for key, value in data.items():
        for key1, value1 in value.items():
            if isinstance(value1, str):
                if value1.find(r"${Model.name}") != -1:
                    value1 = value1.replace(r"${Model.name}", model_name)
                if value1.find(r"${Daytime}") != -1:
                    value1 = value1.replace(r"${Daytime}", date)
                if value1.find(r"${Training.root_path}") != -1:
                    value1 = value1.replace(r"${Training.root_path}", train_root_path)
                if value1.find(r"${Testing.root_path}") != -1:
                    value1 = value1.replace(r"${Testing.root_path}", test_root_path)

                data[key][key1] = value1

            if isinstance(value1, list):
                for id, item in enumerate(value1):
                    if isinstance(item, str):
                        if item.find(r"${Model.name}") != -1:
                            value1[id] = item.replace(r"${Model.name}", model_name)
                        if item.find(r"${Daytime}") != -1:
                            value1[id] = item.replace(r"${Daytime}", date)
                        if item.find(r"${Training.root_path}") != -1:
                            value1[id] = item.replace(r"${Training.root_path}", train_root_path)
                        if item.find(r"${Testing.root_path}") != -1:
                            value1[id] = item.replace(r"${Testing.root_path}", test_root_path)
    if is_debug:
        print("")  # New Line
        print("You can turn off this debug message from yaml_processor(is_debug=False)")

        for key, value in data.items():

            print("===== " + key + " =====")
            for key1, value1 in value.items():
                print(key1 + ":", data[key][key1])
            print("")  # New Line

    return data


if __name__ == "__main__":
    data = yaml_processor(config_file_path="../config/config.yaml", is_debug=True)