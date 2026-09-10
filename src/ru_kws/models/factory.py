from ru_kws.models.bcresnet import BCResNets


def build_model(config, num_classes):
    if config["model"]["name"] != "bcresnet":
        raise ValueError("Unsupported model")
    return BCResNets(base_c=config["model"]["base_c"], num_classes=num_classes)
