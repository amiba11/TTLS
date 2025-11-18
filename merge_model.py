import torch
from recbole.utils import get_model, set_color
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataset import SequentialDataset 

def save_checkpoint(mymodel, config, verbose=True):
    state = {
        "config": config,
        "state_dict": mymodel.state_dict(),
        "other_parameter": mymodel.other_parameter(),
    }
    torch.save(state, f'saved/Mamba4Rec_gowalla_new_model_0.8.pth', pickle_protocol=4)
    if verbose:
        mymodel.logger.info(
            set_color("Saving current", "blue") + f": {'saved/Mamba4Rec_gowalla_new_model_0.8.pth'}"
        )

def merge_model(
    model=None,
    dataset=None,
    config_file_list=None,
):

    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
    )

    dataset = create_dataset(config)  
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 加载训练好的模型的状态字典
    # -- Gowalla --
    # model1_state_dict = torch.load('saved/GRU4Rec-Nov-03-2025_21-07-21.pth', map_location='cuda') # original
    # model2_state_dict = torch.load('', map_location='cuda')
    model1_state_dict = torch.load('saved/Mamba4Rec-Nov-04-2025_20-42-00.pth', map_location='cuda')  # 0.95
    model2_state_dict = torch.load('saved/Mamba4Rec-Nov-04-2025_20-50-36.pth', map_location='cuda')

    # -- Twitch --
    # saved/Mamba4Rec-Nov-05-2025_17-50-35.pth
    # saved/SASRec-Oct-30-2025_23-54-05.pth

    # -- Amazon-Video-Games --
    # model1_state_dict = torch.load('saved/GRU4Rec-Nov-03-2025_16-46-45.pth', map_location='cuda') # 0.8
    # model2_state_dict = torch.load('saved/GRU4Rec-Nov-03-2025_17-02-00.pth', map_location='cuda') # 0.2

    # 定义权重
    weight1 = 0.95
    weight2 = 0.05

    # 创建新的模型实例
    model1 = get_model(model)(config, train_data._dataset)
    model2 = get_model(model)(config, train_data._dataset)

    # 加载状态字典
    model1.load_state_dict(model1_state_dict["state_dict"])
    model1.load_other_parameter(model1_state_dict.get("other_parameter"))
    model2.load_state_dict(model2_state_dict["state_dict"])
    model2.load_other_parameter(model2_state_dict.get("other_parameter"))

    # 创建一个新的模型
    new_model = get_model(model)(config, train_data._dataset)

    # 加权相加
    for param1, param2, new_param in zip(model1.parameters(), model2.parameters(), new_model.parameters()):
        new_param.data = weight1 * param1.data + weight2 * param2.data

    # 保存新模型的状态字典
    save_checkpoint(new_model, config, verbose=True)