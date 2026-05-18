from .Stereo_big_sound_node import Stereo_big_sound_node

NODE_CLASS_MAPPINGS = {
    "BigSound_StereoEnhancer": Stereo_big_sound_node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BigSound_StereoEnhancer": "Stereo Big Sound Enhancer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
