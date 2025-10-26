% Step 0
action(goto(cabinet_1)).
current_position(middle_of_room).
reachable_location(cabinet_1).
reachable_location(cabinet_2).
reachable_location(cabinet_3).
reachable_location(cabinet_4).
reachable_location(cabinet_5).
reachable_location(cabinet_6).
reachable_location(cabinet_7).
reachable_location(cabinet_8).
reachable_location(cabinet_9).
reachable_location(cabinet_10).
reachable_location(cabinet_11).
reachable_location(cabinet_12).
reachable_location(cabinet_13).
reachable_location(coffeemachine_1).
reachable_location(countertop_1).
reachable_location(countertop_2).
reachable_location(diningtable_1).
reachable_location(drawer_1).
reachable_location(drawer_2).
reachable_location(drawer_3).
reachable_location(drawer_4).
reachable_location(fridge_1).
reachable_location(garbagecan_1).
reachable_location(microwave_1).
reachable_location(shelf_1).
reachable_location(shelf_2).
reachable_location(shelf_3).
reachable_location(sinkbasin_1).
reachable_location(stoveburner_1).
reachable_location(stoveburner_2).
reachable_location(stoveburner_3).
reachable_location(stoveburner_4).
reachable_location(toaster_1).

% Step 1
action(goto(fridge_1)).
current_position(cabinet_1).
location_status(cabinet_1, closed).
reachable_location(cabinet_1).
reachable_location(cabinet_2).
reachable_location(cabinet_3).
reachable_location(cabinet_4).
reachable_location(cabinet_5).
reachable_location(cabinet_6).
reachable_location(cabinet_7).
reachable_location(cabinet_8).
reachable_location(cabinet_9).
reachable_location(cabinet_10).
reachable_location(cabinet_11).
reachable_location(cabinet_12).
reachable_location(cabinet_13).
reachable_location(coffeemachine_1).
reachable_location(countertop_1).
reachable_location(countertop_2).
reachable_location(diningtable_1).
reachable_location(drawer_1).
reachable_location(drawer_2).
reachable_location(drawer_3).
reachable_location(drawer_4).
reachable_location(fridge_1).
reachable_location(garbagecan_1).
reachable_location(microwave_1).
reachable_location(shelf_1).
reachable_location(shelf_2).
reachable_location(shelf_3).
reachable_location(sinkbasin_1).
reachable_location(stoveburner_1).
reachable_location(stoveburner_2).
reachable_location(stoveburner_3).
reachable_location(stoveburner_4).
reachable_location(toaster_1).
empty(cabinet_1).
location_status(cabinet_1, closed).

% Step 2
action(open(cabinet_1)).
current_position(fridge_1).
location_status(fridge_1, closed).
reachable_location(cabinet_1).
reachable_location(cabinet_2).
reachable_location(cabinet_3).
reachable_location(cabinet_4).
reachable_location(cabinet_5).
reachable_location(cabinet_6).
reachable_location(cabinet_7).
reachable_location(cabinet_8).
reachable_location(cabinet_9).
reachable_location(cabinet_10).
reachable_location(cabinet_11).
reachable_location(cabinet_12).
reachable_location(cabinet_13).
reachable_location(coffeemachine_1).
reachable_location(countertop_1).
reachable_location(countertop_2).
reachable_location(diningtable_1).
reachable_location(drawer_1).
reachable_location(drawer_2).
reachable_location(drawer_3).
reachable_location(drawer_4).
reachable_location(fridge_1).
reachable_location(garbagecan_1).
reachable_location(microwave_1).
reachable_location(shelf_1).
reachable_location(shelf_2).
reachable_location(shelf_3).
reachable_location(sinkbasin_1).
reachable_location(stoveburner_1).
reachable_location(stoveburner_2).
reachable_location(stoveburner_3).
reachable_location(stoveburner_4).
reachable_location(toaster_1).
empty(cabinet_1).
location_status(cabinet_1, closed).
empty(fridge_1).
location_status(fridge_1, closed).

% Step 3
action(goto(cabinet_1)).
current_position(fridge_1).
location_status(fridge_1, closed).
reachable_location(cabinet_1).
reachable_location(cabinet_2).
reachable_location(cabinet_3).
reachable_location(cabinet_4).
reachable_location(cabinet_5).
reachable_location(cabinet_6).
reachable_location(cabinet_7).
reachable_location(cabinet_8).
reachable_location(cabinet_9).
reachable_location(cabinet_10).
reachable_location(cabinet_11).
reachable_location(cabinet_12).
reachable_location(cabinet_13).
reachable_location(coffeemachine_1).
reachable_location(countertop_1).
reachable_location(countertop_2).
reachable_location(diningtable_1).
reachable_location(drawer_1).
reachable_location(drawer_2).
reachable_location(drawer_3).
reachable_location(drawer_4).
reachable_location(fridge_1).
reachable_location(garbagecan_1).
reachable_location(microwave_1).
reachable_location(shelf_1).
reachable_location(shelf_2).
reachable_location(shelf_3).
reachable_location(sinkbasin_1).
reachable_location(stoveburner_1).
reachable_location(stoveburner_2).
reachable_location(stoveburner_3).
reachable_location(stoveburner_4).
reachable_location(toaster_1).
empty(cabinet_1).
location_status(cabinet_1, closed).
empty(fridge_1).
location_status(fridge_1, closed).

% Step 4
action(open(fridge_1)).
current_position(cabinet_1).
location_status(cabinet_1, closed).
reachable_location(cabinet_1).
reachable_location(cabinet_2).
reachable_location(cabinet_3).
reachable_location(cabinet_4).
reachable_location(cabinet_5).
reachable_location(cabinet_6).
reachable_location(cabinet_7).
reachable_location(cabinet_8).
reachable_location(cabinet_9).
reachable_location(cabinet_10).
reachable_location(cabinet_11).
reachable_location(cabinet_12).
reachable_location(cabinet_13).
reachable_location(coffeemachine_1).
reachable_location(countertop_1).
reachable_location(countertop_2).
reachable_location(diningtable_1).
reachable_location(drawer_1).
reachable_location(drawer_2).
reachable_location(drawer_3).
reachable_location(drawer_4).
reachable_location(fridge_1).
reachable_location(garbagecan_1).
reachable_location(microwave_1).
reachable_location(shelf_1).
reachable_location(shelf_2).
reachable_location(shelf_3).
reachable_location(sinkbasin_1).
reachable_location(stoveburner_1).
reachable_location(stoveburner_2).
reachable_location(stoveburner_3).
reachable_location(stoveburner_4).
reachable_location(toaster_1).
empty(cabinet_1).
location_status(cabinet_1, closed).
empty(fridge_1).
location_status(fridge_1, closed).
