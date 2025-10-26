action_failed(put(Item, Receptacle)) :-
    item_in_location(Item, ItemLoc),
    item_in_location(Receptacle, ReceptacleLoc),
    ItemLoc \= ReceptacleLoc.

action_failed(put(Item, Receptacle)) :-
    items_in_location(Receptacle, _),
    location_status(Receptacle, full).