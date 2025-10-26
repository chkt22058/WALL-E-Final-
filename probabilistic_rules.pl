% Probabilistic Prolog Rules
% Format: p :: rule
% p = False / (True + False)

% No data for this rule (condition never satisfied)
% action_failed(put(Item, Destination)) :- \+ reachable_location(Destination).

% No data for this rule (condition never satisfied)
% action_failed(take(Item, Location)) :- current_position(Location), \+ items_in_location(Item, Location).

% Statistics: Success=True: 0, Success=False: 17, Total: 17
0.5 :: action_failed(take(Item, Location)) :- current_position(CurrLoc), \+ items_in_location(Item, CurrLoc).

% No data for this rule (condition never satisfied)
% action_failed(put(Item, Receptacle)) :- current_position(CurrLoc), \+ items_in_location(Item, CurrLoc).

% No data for this rule (condition never satisfied)
% action_failed(open(Receptacle)) :- current_position(CurrLoc), \+ items_in_location(Receptacle, CurrLoc).

% No data for this rule (condition never satisfied)
% action_failed(close(Receptacle)) :- current_position(CurrLoc), \+ items_in_location(Receptacle, CurrLoc).

% No data for this rule (condition never satisfied)
% action_failed(clean(Item, Location)) :- current_position(CurrLoc), \+ items_in_location(Item, CurrLoc).

% No data for this rule (condition never satisfied)
% action_failed(heat(Item, Location)) :- current_position(CurrLoc), \+ items_in_location(Item, CurrLoc).

% No data for this rule (condition never satisfied)
% action_failed(cool(Item, Location)) :- current_position(CurrLoc), \+ items_in_location(Item, CurrLoc).

% No data for this rule (condition never satisfied)
% action_failed(use(Tool)) :- current_position(CurrLoc), \+ items_in_location(Tool, CurrLoc).

% No data for this rule (condition never satisfied)
% action_failed(goto(Destination)) :- \+ reachable_location(Destination).

% Statistics: Success=True: 0, Success=False: 9, Total: 9
0.5 :: action_failed(take(Item, Location)) :- \+ items_in_location(Item, Location).

% No data for this rule (condition never satisfied)
% action_failed(take(Item, Location)) :- items_in_location(Item, Location), \+ reachable_location(Location).

% No data for this rule (condition never satisfied)
% action_failed(take(Item, Location)) :- item_in_hand(Item, in_hand).

% No data for this rule (condition never satisfied)
% action_failed(take(Item, Location)) :- \+ reachable_location(Location).

% Statistics: Success=True: 0, Success=False: 17, Total: 17
0.5 :: action_failed(take(Item, Location)) :- \+ empty(Location).

% No data for this rule (condition never satisfied)
% (no output)

% No data for this rule (condition never satisfied)
% action_failed(take(Item, Location)) :- item_in_hand(Item, held).

% No data for this rule (condition never satisfied)
% action_failed(take(Item, Location)) :- item_in_hand(_, in_hand).

% Statistics: Success=True: 0, Success=False: 17, Total: 17
0.5 :: action_failed(take(Item, Receptacle)) :- current_position(Position), Position \== Receptacle.

% No data for this rule (condition never satisfied)
% action_failed(take(Item, Receptacle)) :- \+ reachable_location(Receptacle).

% Statistics: Success=True: 0, Success=False: 9, Total: 9
0.5 :: action_failed(take(Item, Receptacle)) :- \+ items_in_location(Item, Receptacle).

% No data for this rule (condition never satisfied)
% action_failed(use(Item)) :- item_in_hand(_, in_hand), Item \== _.

% No data for this rule (condition never satisfied)
% action_failed(put(Item, Receptacle)) :- \+ empty(Receptacle).

% No data for this rule (condition never satisfied)
% action_failed(take(Item, Receptacle)) :- items_in_location(Item, Receptacle), \+ reachable_location(Receptacle).

% Statistics: Success=True: 0, Success=False: 17, Total: 17
0.5 :: action_failed(take(Item, Receptacle)) :- \+ empty(Receptacle).

% No data for this rule (condition never satisfied)
% action_failed(take(Item, Location)) :- item_in_hand(_, holding).

