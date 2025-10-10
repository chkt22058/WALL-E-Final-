def generate_action_result_from_obs(obs_text):
    """
    観測テキストに基づいて action_result を作成する

    Args:
        obs_text (str): 環境からの観測（自然言語）

    Returns:
        dict: action_result（feedback, success, suggestion=""）
    """
    fail_phrases = [
        "nothing happens"
    ]

    obs_text_lower = obs_text.lower()
    success = True

    for phrase in fail_phrases:
        if phrase in obs_text_lower:
            success = False
            break


    return {
        "feedback": obs_text.strip(),
        "success": success,
        "suggestion": ""
    }
