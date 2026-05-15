import hashlib


MAX_SEED = 2**32 - 1
SEED_RANGE = MAX_SEED + 1


class XJImageToSeed:
    """
    Generates a deterministic seed from image pixel data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "offset": (
                    "INT",
                    {"default": 0, "min": 0, "max": MAX_SEED, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate_seed"
    CATEGORY = "XJNodes/image"

    def generate_seed(self, image, offset):
        # Convert image tensor to numpy array
        # image shape: (B, H, W, C) where B is batch size
        img_np = image.cpu().numpy()

        # Create a hash from the image data
        img_bytes = img_np.tobytes()
        hash_value = hashlib.md5(img_bytes).hexdigest()

        # Convert hash to integer and add offset
        seed = int(hash_value, 16) % SEED_RANGE
        seed = (seed + offset) % SEED_RANGE

        return (seed,)


NODE_CLASS_MAPPINGS = {
    "XJImageToSeed": XJImageToSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJImageToSeed": "Image to Seed",
}
