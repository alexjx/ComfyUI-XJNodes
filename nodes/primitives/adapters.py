import comfy.samplers


class XJSchedulerAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "defaultInput": False,
                    },
                ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    CATEGORY = "XJNode/Util"
    FUNCTION = "doit"

    def doit(self, scheduler):
        return (scheduler,)


class XJSamplerAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "defaultInput": False,
                    },
                ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    CATEGORY = "XJNode/Util"
    FUNCTION = "doit"

    def doit(self, sampler):
        return (sampler,)


class XJStringPass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "string": (
                    "STRING",
                    {"forceInput": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "passthrough"
    CATEGORY = "XJNode/Util"
    DESCRIPTION = """
Passes the string through without modifying it.
"""

    def passthrough(self, string=None):
        return (string,)


NODE_CLASS_MAPPINGS = {
    "XJSchedulerAdapter": XJSchedulerAdapter,
    "XJSamplerAdapter": XJSamplerAdapter,
    "XJStringPass": XJStringPass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSchedulerAdapter": "Scheduler Adapter",
    "XJSamplerAdapter": "Sampler Adapter",
    "XJStringPass": "StringPass",
}
