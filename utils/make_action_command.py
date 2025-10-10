def make_action_command(response: dict) -> str:
    """
    Convert LLMAgent response into action command string.
    
    Example:
    {
        "action_name": "put",
        "args": {"obj": "alarmclock", "target": "desk"}
    }
    -> "put alarmclock in desk"
    """
    action = response.get("action_name")
    args = response.get("args", {})

    obj = args.get("obj")
    tool = args.get("tool")
    recep = args.get("recep")

    if action == "goto" or action == "go to":
        return f"go to {recep}"
    elif action == "take":
        return f"take {obj} from {recep}"
    elif action == "put":
        return f"move {obj} to {recep}"
    elif action == "open":
        return f"open {recep}"
    elif action == "close":
        return f"close {recep}"
    elif action == "clean":
        return f"clean {obj} with {recep}"
    elif action == "heat":
        return f"heat {obj} with {recep}"
    elif action == "cool":
        return f"cool {obj} with {recep}"
    elif action == "use":
        return f"use {tool}"
    else:
        raise ValueError(f"Unknown action: {action}")
