def check_is_reachable(state: dict, action: dict) -> bool:
    target_location = None
    action_name = action.get("action_name")
    args = action.get("args", {})
    if action_name in ["goto", "go to"]:
        target_location = args.get("recep") or args.get("location")
    if not target_location:
        return False
    reachable_locations = state.get("reachable_locations", [])
    return target_location in reachable_locations

def check_is_unreachable(state: dict, action: dict) -> bool:
    target_location = None
    action_name = action.get("action_name")
    args = action.get("args", {})
    if action_name in ["goto", "go to"]:
        target_location = args.get("recep") or args.get("location")
    if not target_location:
        return False
    reachable_locations = state.get("reachable_locations", [])
    return target_location not in reachable_locations