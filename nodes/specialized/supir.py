class XJSupirParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": (
                    "INT",
                    {
                        "forceInput": True,
                    },
                ),
                "iterations": (
                    "INT",
                    {
                        "forceInput": True,
                    },
                ),
                "final_control_scale_start": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01},
                ),
                "final_control_scale_end": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 2.00, "step": 0.01},
                ),
                "control_scale_step": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.00, "max": 1.00, "step": 0.01},
                ),
                "final_s_noise": (
                    "FLOAT",
                    {"default": 1.003, "min": 1.000, "max": 1.100, "step": 0.001},
                ),
                "s_noise_step": (
                    "FLOAT",
                    {"default": 0.001, "min": -0.100, "max": 0.100, "step": 0.001},
                ),
                "final_EDM_s_churn": (
                    "INT",
                    {"default": 3, "min": 0, "max": 40, "step": 1},
                ),
                "EDM_s_churn_step": (
                    "INT",
                    {"default": 1, "min": -40, "max": 40, "step": 1},
                ),
            },
        }

    RETURN_NAMES = (
        "control_scale_start",
        "control_scale_end",
        "s_noise",
        "EDM_s_churn",
    )
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "INT")
    FUNCTION = "calculate"

    def calculate(
        self,
        index,
        iterations,
        final_control_scale_start,
        final_control_scale_end,
        control_scale_step,
        final_s_noise,
        s_noise_step,
        final_EDM_s_churn,
        EDM_s_churn_step,
    ):
        offset = iterations - index - 1
        if offset < 0:
            offset = 0
        # Calculate the values based on the offset
        control_scale_start = final_control_scale_start - (control_scale_step * offset)
        if control_scale_start < 0:
            control_scale_start = 0
        control_scale_end = final_control_scale_end - (control_scale_step * offset)
        if control_scale_end < 0:
            control_scale_end = 0
        s_noise = final_s_noise - (s_noise_step * offset)
        if s_noise_step > 0:
            if s_noise < 1.0:
                s_noise = 1.0
        elif s_noise_step < 0:
            if s_noise > 1.1:
                s_noise = 1.1
        EDM_s_churn = final_EDM_s_churn - (EDM_s_churn_step * offset)
        if EDM_s_churn_step > 0:
            if EDM_s_churn < 0:
                EDM_s_churn = 0
        elif EDM_s_churn_step < 0:
            if EDM_s_churn > 40:
                EDM_s_churn = 40
        return (control_scale_start, control_scale_end, s_noise, EDM_s_churn)


NODE_CLASS_MAPPINGS = {
    "XJSupirParameters": XJSupirParameters,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XJSupirParameters": "SUPIR Parameters",
}
