# Pruned code rules

def Rule_10_goto(state, action, scene_graph):
    # Only apply to 'goto' actions
    if action.get("action_name") != "goto":
        return "Action type not applicable to this rule.", True, ""
    recep = action.get("args", {}).get("recep")
    current_location = state.get("current_position", {}).get("location_name")
    current_location_info = state.get("items_in_locations", {}).get(current_location, {})
    adjacent = current_location_info.get("adjacent", [])
    if recep in adjacent:
        feedback = f"Failed: The target location '{recep}' is already adjacent to your current location '{current_location}'."
        success = False
        suggestion = f"Try moving to a non-adjacent location or interact with '{recep}' directly."
        return feedback, success, suggestion
    feedback = f"Success: The target location '{recep}' is not adjacent to your current location."
    return feedback, True, ""

def Rule_17_heat(state, action, scene_graph):
    # Rule 17: For action heat, fail if obj is not in recep's items before action
    if action.get("action_name") != "heat":
        return "Action type does not match this rule, passing.", True, ""
    obj = action["args"].get("obj")
    recep = action["args"].get("recep")
    items_in_locations = state.get("items_in_locations", {})
    if recep not in items_in_locations:
        return f"Target location '{recep}' not found in current items_in_locations.", False, f"Move to or specify a valid heating appliance or location."
    recep_items = items_in_locations[recep].get("items", [])
    if obj not in recep_items:
        return f"Cannot heat '{obj}' because it is not placed in or on '{recep}'.", False, f"Place '{obj}' in or on '{recep}' before heating."
    return f"'{obj}' is correctly placed in or on '{recep}' for heating.", True, ""

