from timm.models import efficientnet_lite0, load_pretrained, register_model


@register_model
def custom_efficientnet_lite0(pretrained=False, **kwargs):
    model = efficientnet_lite0(
        pretrained=False,
        **kwargs,
    )

    # TODO: customize the model here

    # update model.pretrained_cfg
    pretrained_cfg = model.pretrained_cfg
    # {
    #    "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth",
    #    "hf_hub_id": "timm/efficientnet_lite0.ra_in1k",
    #    "architecture": "efficientnet_lite0",
    #    "tag": "ra_in1k",
    #    "custom_load": False,
    #    "input_size": (3, 224, 224),
    #    "fixed_input_size": False,
    #    "interpolation": "bicubic",
    #    "crop_pct": 0.875,
    #    "crop_mode": "center",
    #    "mean": (0.485, 0.456, 0.406),
    #    "std": (0.229, 0.224, 0.225),
    #    "num_classes": 1000,
    #    "pool_size": (7, 7),
    #    "first_conv": "conv_stem",
    #    "classifier": "classifier",
    # }
    pretrained_cfg["url"] = None
    pretrained_cfg["hf_hub_id"] = None
    pretrained_cfg["architecture"] = "custom_efficientnet_lite0"
    pretrained_cfg["tag"] = None
    pretrained_cfg["file"] = "/home/motoki_kimura/tmp/timm_efficientnet_lite0.pth"  # TODO: use hf hub instead
    model.pretrained_cfg = pretrained_cfg

    if pretrained:
        # Load the weight file specified by model.pretrained_cfg["file"]
        # If `in_chans`` or `num_classes` are specified, the weight is modified before loading
        load_pretrained(
            model,
            pretrained_cfg=None,
            in_chans=kwargs.get("in_chans", 3),
            num_classes=kwargs.get("num_classes", 1000),
            filter_fn=None,
            strict=True,
        )

    return model
