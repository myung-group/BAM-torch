from .ga_forward import model_forward, pa_model_forward, base_foward


FORWARD_REGISTRY = {
    "prob": pa_model_forward,
    "probabilistic": pa_model_forward,
    "stochastic": model_forward,
    "det": model_forward,
    "se3-stochastic": model_forward,
    "se3-det": model_forward,
    "se3-all": model_forward,
    "stochastic": model_forward,
    "no": base_foward,
}