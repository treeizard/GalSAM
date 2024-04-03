from base_dataset import RadioGalaxyNET

class lora_dataset(RadioGalaxyNET):
    def __init__ (self, root: str, annFile: str, transform=None, transforms=None):
        