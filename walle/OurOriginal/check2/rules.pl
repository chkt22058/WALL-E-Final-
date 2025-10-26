action_failed(goto(Destination)) :- empty(Destination).
action_failed(goto(Destination)) :- location_status(Destination, Status), (Status = locked; Status = blocked).
action_failed(goto(Destination)) :- \+ reachable_location(Destination).
action_failed(goto(Destination)) :- location_status(Destination, Status), Status \== clear.
action_failed(open(Location)) :- location_status(Location, open).
action_failed(goto(Destination)) :- current_position(Current), Destination == Current.
action_failed(open(Receptacle)) :- location_status(Receptacle, open).
action_failed(open(Receptacle)) :- item_in_hand(Item, not_held).
action_failed(open(Receptacle)) :- empty(Receptacle).
