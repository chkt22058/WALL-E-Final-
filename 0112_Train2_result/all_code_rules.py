# Pruned code rules

def Rule_61_put(state, action, scene_graph):
    if action.get("action_name") != "put":
        return "Action type does not match rule, skipping.", True, ""
    current_loc = state.get("current_position", {}).get("location_name")
    recep = action.get("args", {}).get("recep")
    if current_loc != recep:
        feedback = f"Failed: Agent is at '{current_loc}', but needs to be at '{recep}' to put the object."
        suggestion = f"Go to '{recep}' before attempting to put the object."
        return feedback, False, suggestion
    return "Success: Agent is at the correct location to put the object.", True, ""

def Rule_34_goto(state, action, scene_graph):
    feedback = ""
    success = True
    suggestion = ""
    if action.get("action_name") != "goto":
        return "Action type not applicable for this rule.", True, ""
    target = action.get("args", {}).get("recep")
    current_location = state.get("current_position", {}).get("location_name")
    # Get adjacent locations for current location
    items_in_locations = state.get("items_in_locations", {})
    adjacent = []
    if current_location in items_in_locations:
        adjacent = items_in_locations[current_location].get("adjacent", [])
    # Check if target is adjacent
    if target in adjacent:
        # Check if agent's position does not change (implies movement failed)
        # Find agent's current location in scene_graph
        agent_location = None
        for edge in scene_graph.get("edges", []):
            if edge.get("source") == "agent" and edge.get("relation") == "at":
                agent_location = edge.get("target")
                break
        # If agent's location is still the same as current_location, movement failed
        if agent_location == current_location:
            feedback = f"Failed to move to {target}: Environmental constraint prevents movement despite adjacency."
            success = False
            suggestion = f"Try a different route or check for obstacles between {current_location} and {target}."
        else:
            feedback = f"Moved to {target} successfully."
            success = True
            suggestion = ""
    else:
        feedback = f"{target} is not adjacent to {current_location}, rule not applicable."
        success = True
        suggestion = ""
    return feedback, success, suggestion

def Rule_2003_cool(state, action, scene_graph):
    # Only apply to 'cool' actions
    if action.get("action_name") != "cool":
        return "Action type not applicable to this rule.", True, ""
    obj = action["args"].get("obj")
    recep = action["args"].get("recep")
    agent_loc = state.get("current_position", {}).get("location_name")
    item_in_hand = state.get("item_in_hand", {}).get("item_name")
    # If agent is not at recep and is holding obj, cannot place obj into recep
    if agent_loc != recep and item_in_hand == obj:
        feedback = f"Failed: You are not at {recep} and are still holding {obj}, so you cannot place it into {recep} to cool."
        success = False
        suggestion = f"Go to {recep} and place {obj} into it before cooling."
        return feedback, success, suggestion
    feedback = f"Success: Agent is at {recep} or can place {obj} into {recep}."
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

def Rule_10_cool(state, action, scene_graph):
    # Rule 10: For action cool, fail if the object is not in a state/context where cooling is meaningful.
    if action.get("action_name") != "cool":
        return "Action type does not match rule; skipping.", True, ""
    obj = action["args"].get("obj")
    recep = action["args"].get("recep")
    if not obj:
        return "No object specified to cool.", False, "Specify an object to cool."
    # Define coolable object types
    coolable_keywords = ["food", "drink", "can", "bottle", "fruit", "vegetable", "milk", "juice", "water", "soda", "egg", "meat", "icecream", "yogurt", "cheese", "leftover", "cake", "pie", "dessert", "soup", "beverage"]
    obj_lower = obj.lower()
    is_coolable = any(word in obj_lower for word in coolable_keywords)
    # Define cooling appliances
    cooling_appliances = ["fridge", "freezer", "refrigerator", "cooler"]
    recep_lower = (recep or "").lower()
    is_in_cooling_appliance = any(appliance in recep_lower for appliance in cooling_appliances)
    # Check if object is in a cooling appliance (via scene_graph)
    in_cooling_appliance = False
    for edge in scene_graph.get("edges", []):
        if edge["relation"] == "contains" and edge["target"] == obj:
            for appliance in cooling_appliances:
                if appliance in edge["source"]:
                    in_cooling_appliance = True
                    break
    if not (is_coolable or is_in_cooling_appliance or in_cooling_appliance):
        feedback = f"The object '{obj}' is not in a context where cooling is meaningful."
        suggestion = f"Try cooling a food or drink item, or place the object in a fridge or freezer before cooling."
        return feedback, False, suggestion
    return f"The object '{obj}' is in a context where cooling is meaningful.", True, ""

