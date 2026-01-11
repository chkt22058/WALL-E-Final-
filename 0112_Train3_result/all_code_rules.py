# Pruned code rules

def Rule_30003_goto(state, action, scene_graph):
    # Rule: goto fails if target is current location or is adjacent and agent is already facing it
    if action.get("action_name") != "goto":
        return "Action type does not match this rule.", True, ""
    target = action.get("args", {}).get("recep")
    current_location = state.get("current_position", {}).get("location_name")
    adjacent = []
    if current_location in state.get("items_in_locations", {}):
        adjacent = state["items_in_locations"][current_location].get("adjacent", [])
    if target == current_location:
        feedback = f"Failed: You are already at '{target}'."
        success = False
        suggestion = "Try moving to a different location."
        return feedback, success, suggestion
    if target in adjacent:
        feedback = f"Failed: '{target}' is directly adjacent and you are already facing it. No movement occurs."
        success = False
        suggestion = "Try moving to a non-adjacent location or interact with the adjacent location."
        return feedback, success, suggestion
    return "Success: The target location is not the current or directly adjacent location.", True, ""

def Rule_61_cool(state, action, scene_graph):
    # Rule 61: For action cool, fail if agent not at cooling device location
    if action.get("action_name") != "cool":
        return "Action type does not match this rule. No check needed.", True, ""
    recep = action.get("args", {}).get("recep")
    current_location = state.get("current_position", {}).get("location_name")
    if current_location != recep:
        feedback = f"Failed: You are not at the location of the cooling device ('{recep}')."
        suggestion = f"Go to '{recep}' before attempting to cool."
        return feedback, False, suggestion
    return "Success: You are at the cooling device location.", True, ""

def Rule_22_put(state, action, scene_graph):
    # Applies to 'put' and 'heat' actions
    if action["action_name"] not in ["put", "heat"]:
        return "Action type not relevant to this rule.", True, ""
    recep = action["args"].get("recep")
    agent_loc = state.get("current_position", {}).get("location_name")
    if not recep or not agent_loc:
        return "Missing information about agent location or target receptacle.", False, "Check that both agent location and target receptacle are specified."
    # Check if agent is at or adjacent to the receptacle
    if recep == agent_loc:
        return "Agent is at the target receptacle.", True, ""
    # Check adjacency
    agent_loc_info = state.get("items_in_locations", {}).get(agent_loc, {})
    adjacents = agent_loc_info.get("adjacent", [])
    if recep in adjacents:
        return "Agent is adjacent to the target receptacle.", True, ""
    return f"Agent is not at or adjacent to '{recep}'.", False, f"Move to '{recep}' or an adjacent location before performing this action."

def Rule_23_open(state, action, scene_graph):
    recep = action.get("args", {}).get("recep", None)
    if action.get("action_name") != "open":
        return "Action type does not match rule; skipping rule.", True, ""
    if not recep:
        return "No target specified to open.", False, "Specify a valid object or location to open."
    # Check if recep supports open/close (status field ever changes)
    recep_status = None
    if "items_in_locations" in state and recep in state["items_in_locations"]:
        recep_status = state["items_in_locations"][recep].get("status", None)
    elif "current_position" in state and state["current_position"].get("location_name") == recep:
        recep_status = state["current_position"].get("status", None)
    # If status is always None or missing, it does not support open/close
    if recep_status is None:
        return f"Cannot open {recep}: it does not support being opened or closed.", False, f"Choose a container or device that can be opened, not {recep}."
    return f"Open action on {recep} is allowed.", True, ""

def Rule_46_cool(state, action, scene_graph):
    # Rule 46: Fail if neither the object nor the receiving location is in a state that allows for temperature transfer.
    if action.get("action_name") != "cool":
        return "Action type does not match rule; skipping.", True, ""
    obj = action["args"].get("obj")
    recep = action["args"].get("recep")
    # We'll check if both object and recep are in neutral/inactive state (not hot, not cold, not active)
    # Heuristic: If status is missing or 'neutral', 'room temperature', 'inactive', etc.
    neutral_statuses = ["neutral", "room temperature", "inactive", "off", None]
    # Check object status
    obj_status = None
    for loc, locinfo in state.get("items_in_locations", {}).items():
        if obj in locinfo.get("items", []):
            obj_status = locinfo.get("status", None)
            break
    if state.get("item_in_hand", {}).get("item_name") == obj:
        obj_status = state.get("item_in_hand", {}).get("status", None)
    # Check recep status
    recep_status = None
    if recep in state.get("items_in_locations", {}):
        recep_status = state["items_in_locations"][recep].get("status", None)
    # If both are neutral/inactive, fail
    if (obj_status in neutral_statuses or obj_status is None) and (recep_status in neutral_statuses or recep_status is None):
        feedback = f"Neither '{obj}' nor '{recep}' is in a state that allows for temperature transfer."
        suggestion = f"Try cooling an object that is hot, or use a receiving location that is actively cooling."
        return feedback, False, suggestion
    return f"Temperature transfer is possible between '{obj}' and '{recep}'.", True, ""

