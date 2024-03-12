from . import APIHybridizationProbability, APIBase

def generare_api(ai_filter: str, ai_filter_path: str = None) -> APIBase:
    """Initializeds the correct api class for the desided ai filter.

    :param ai_filter: Type of AI filter to use. {hybridization_probabaility}
    :type ai_filter: str
    :param ai_filter_path: Path to the AI filter. If not provided, the default path will be used. Defaults to None
    :type ai_filter_path: str
    :return: API class for the AI filter.
    :rtype: APIBase 
    """

    if ai_filter == "hybridization_probabaility":
        return APIHybridizationProbability(ai_filter_path=ai_filter_path)
    else:
        raise ValueError(f"The AI filter {ai_filter} is not recognized.")