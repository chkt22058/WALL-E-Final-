action_failed(goto(Destination)) :- reachable_location(Destination), \+ reachable_location(Destination).
action_failed(take(Item, _)) :- \+ (reachable_location(Location), items_in_location(Item, Location)).
action_failed(take(Item, Location)) :- \+ items_in_location(Item, Location).
action_failed(put(Item, Receptacle)) :- \+ items_in_location(Item, Receptacle).
action_failed(open(Receptacle)) :- location_status(Receptacle, Status), \+ allowed_status(open, Status).
action_failed(close(Receptacle)) :- location_status(Receptacle, Status), \+ allowed_status(close, Status).
action_failed(clean(Item, Location)) :- location_status(Location, Status), \+ allowed_status(clean, Status).
action_failed(heat(Item, Location)) :- location_status(Location, Status), \+ allowed_status(heat, Status).
action_failed(cool(Item, Location)) :- location_status(Location, Status), \+ allowed_status(cool, Status).
action_failed(use(Tool)) :- item_in_hand(Tool, Status), \+ allowed_status(use, Status).
action_failed(goto(TargetLocation)) :- \+ reachable_location(TargetLocation).
