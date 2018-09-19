# pylint: disable=too-many-lines
import json

from allennlp.semparse.contexts.atis_tables import * # pylint: disable=wildcard-import,unused-wildcard-import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.atis_world import AtisWorld

class TestAtisWorld(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        test_filename = self.FIXTURES_ROOT / "data" / "atis" / "sample.json"
        self.data = open(test_filename).readlines()
        self.database_directory = self.FIXTURES_ROOT / "data" / "atis" / "atis.db"

    def test_atis_global_actions(self): # pylint: disable=no-self-use
        world = AtisWorld(utterances=[],
                          database_directory=str(self.database_directory))
        valid_actions = world.valid_actions
        assert set(valid_actions.keys()) == {'agg',
                                             'agg_func',
                                             'agg_results',
                                             'aircraft_basic_type_string',
                                             'aircraft_manufacturer_string',
                                             'airline_airline_code_string',
                                             'airline_airline_name_string',
                                             'airport_airport_code_string',
                                             'biexpr',
                                             'binaryop',
                                             'boolean',
                                             'city_city_name_string',
                                             'city_city_code_string',
                                             'city_state_code_string',
                                             'class_of_service_booking_class_string',
                                             'col_ref',
                                             'col_refs',
                                             'condition',
                                             'conditions',
                                             'conj',
                                             'days_day_name_string',
                                             'distinct',
                                             'fare_basis_class_type_string',
                                             'fare_basis_economy_string',
                                             'fare_basis_fare_basis_code_string',
                                             'fare_fare_basis_code_string',
                                             'fare_round_trip_required_string',
                                             'flight_airline_code_string',
                                             'flight_flight_days_string',
                                             'flight_flight_number_string',
                                             'flight_stop_stop_airport_string',
                                             'food_service_meal_description_string',
                                             'ground_service_transport_type_string',
                                             'in_clause',
                                             'number',
                                             'pos_value',
                                             'query',
                                             'restriction_restriction_code_string',
                                             'select_results',
                                             'state_state_name_string',
                                             'statement',
                                             'table_name',
                                             'table_refs',
                                             'ternaryexpr',
                                             'time_range_end',
                                             'time_range_start',
                                             'value',
                                             'where_clause'}

        assert set(valid_actions['statement']) == {'statement -> [query, ";"]'}
        assert set(valid_actions['query']) == \
                {'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'query -> ["SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause]'}
        assert set(valid_actions['select_results']) == \
                {'select_results -> [agg]', 'select_results -> [col_refs]'}
        assert set(valid_actions['agg']) == \
                {'agg -> [agg_func, "(", col_ref, ")"]'}
        assert set(valid_actions['agg_func']) == \
                {'agg_func -> ["COUNT"]',
                 'agg_func -> ["MAX"]',
                 'agg_func -> ["MIN"]'}
        assert set(valid_actions['col_refs']) == \
                {'col_refs -> [col_ref]', 'col_refs -> [col_ref, ",", col_refs]'}
        assert set(valid_actions['table_refs']) == \
                {'table_refs -> [table_name]', 'table_refs -> [table_name, ",", table_refs]'}
        assert set(valid_actions['where_clause']) == \
                {'where_clause -> ["WHERE", "(", conditions, ")"]',
                 'where_clause -> ["WHERE", conditions]'}
        assert set(valid_actions['conditions']) == \
                {'conditions -> ["(", conditions, ")", conj, conditions]',
                 'conditions -> ["(", conditions, ")"]',
                 'conditions -> ["NOT", conditions]',
                 'conditions -> [condition, conj, "(", conditions, ")"]',
                 'conditions -> [condition, conj, conditions]',
                 'conditions -> [condition]'}
        assert set(valid_actions['condition']) == \
                {'condition -> [biexpr]',
                 'condition -> [in_clause]',
                 'condition -> [ternaryexpr]'}
        assert set(valid_actions['in_clause']) == \
                {'in_clause -> [col_ref, "IN", query]'}
        assert set(valid_actions['biexpr']) == \
                {'biexpr -> ["aircraft", ".", "basic_type", binaryop, '
                 'aircraft_basic_type_string]',
                 'biexpr -> ["aircraft", ".", "manufacturer", binaryop, '
                 'aircraft_manufacturer_string]',
                 'biexpr -> ["airline", ".", "airline_code", binaryop, '
                 'airline_airline_code_string]',
                 'biexpr -> ["airline", ".", "airline_name", binaryop, '
                 'airline_airline_name_string]',
                 'biexpr -> ["airport", ".", "airport_code", binaryop, '
                 'airport_airport_code_string]',
                 'biexpr -> ["city", ".", "city_code", binaryop, city_city_code_string]',
                 'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                 'biexpr -> ["city", ".", "state_code", binaryop, city_state_code_string]',
                 'biexpr -> ["class_of_service", ".", "booking_class", binaryop, '
                 'class_of_service_booking_class_string]',
                 'biexpr -> ["days", ".", "day_name", binaryop, days_day_name_string]',
                 'biexpr -> ["fare", ".", "fare_basis_code", binaryop, '
                 'fare_fare_basis_code_string]',
                 'biexpr -> ["fare", ".", "round_trip_required", binaryop, '
                 'fare_round_trip_required_string]',
                 'biexpr -> ["fare_basis", ".", "class_type", binaryop, '
                 'fare_basis_class_type_string]',
                 'biexpr -> ["fare_basis", ".", "economy", binaryop, '
                 'fare_basis_economy_string]',
                 'biexpr -> ["fare_basis", ".", "fare_basis_code", binaryop, '
                 'fare_basis_fare_basis_code_string]',
                 'biexpr -> ["flight", ".", "airline_code", binaryop, '
                 'flight_airline_code_string]',
                 'biexpr -> ["flight", ".", "flight_days", binaryop, '
                 'flight_flight_days_string]',
                 'biexpr -> ["flight", ".", "flight_number", binaryop, '
                 'flight_flight_number_string]',
                 'biexpr -> ["flight_stop", ".", "stop_airport", binaryop, '
                 'flight_stop_stop_airport_string]',
                 'biexpr -> ["food_service", ".", "meal_description", binaryop, '
                 'food_service_meal_description_string]',
                 'biexpr -> ["ground_service", ".", "transport_type", binaryop, '
                 'ground_service_transport_type_string]',
                 'biexpr -> ["restriction", ".", "restriction_code", binaryop, '
                 'restriction_restriction_code_string]',
                 'biexpr -> ["state", ".", "state_name", binaryop, state_state_name_string]',
                 'biexpr -> [col_ref, binaryop, value]',
                 'biexpr -> [value, binaryop, value]'}
        assert set(valid_actions['binaryop']) == \
                {'binaryop -> ["*"]',
                 'binaryop -> ["+"]',
                 'binaryop -> ["-"]',
                 'binaryop -> ["/"]',
                 'binaryop -> ["<"]',
                 'binaryop -> ["<="]',
                 'binaryop -> ["="]',
                 'binaryop -> [">"]',
                 'binaryop -> [">="]',
                 'binaryop -> ["IS"]'}
        assert set(valid_actions['ternaryexpr']) == \
                {'ternaryexpr -> [col_ref, "BETWEEN", time_range_start, "AND", time_range_end]',
                 'ternaryexpr -> [col_ref, "NOT", "BETWEEN", time_range_start, "AND", '
                 'time_range_end]'}
        assert set(valid_actions['value']) == \
                {'value -> ["NOT", pos_value]',
                 'value -> [pos_value]'}
        assert set(valid_actions['pos_value']) == \
                {'pos_value -> ["ALL", query]',
                 'pos_value -> ["ANY", query]',
                 'pos_value -> ["NULL"]',
                 'pos_value -> [agg_results]',
                 'pos_value -> [boolean]',
                 'pos_value -> [col_ref]',
                 'pos_value -> [number]'}
        assert set(valid_actions['agg_results']) == \
                {('agg_results -> ["(", "SELECT", distinct, agg, "FROM", table_name, '
                  'where_clause, ")"]'),
                 'agg_results -> ["SELECT", distinct, agg, "FROM", table_name, where_clause]'}
        assert set(valid_actions['boolean']) == \
                {'boolean -> ["true"]', 'boolean -> ["false"]'}
        assert set(valid_actions['conj']) == \
                {'conj -> ["OR"]', 'conj -> ["AND"]'}
        assert set(valid_actions['distinct']) == \
                {'distinct -> [""]', 'distinct -> ["DISTINCT"]'}
        assert set(valid_actions['number']) == \
                {'number -> ["0"]',
                 'number -> ["1"]'}
        assert set(valid_actions['col_ref']) == \
                {'col_ref -> ["*"]',
                 'col_ref -> ["aircraft", ".", "aircraft_code"]',
                 'col_ref -> ["aircraft", ".", "aircraft_description"]',
                 'col_ref -> ["aircraft", ".", "basic_type"]',
                 'col_ref -> ["aircraft", ".", "capacity"]',
                 'col_ref -> ["aircraft", ".", "manufacturer"]',
                 'col_ref -> ["aircraft", ".", "pressurized"]',
                 'col_ref -> ["aircraft", ".", "propulsion"]',
                 'col_ref -> ["aircraft", ".", "wide_body"]',
                 'col_ref -> ["airline", ".", "airline_code"]',
                 'col_ref -> ["airline", ".", "airline_name"]',
                 'col_ref -> ["airport", ".", "airport_code"]',
                 'col_ref -> ["airport", ".", "airport_location"]',
                 'col_ref -> ["airport", ".", "airport_name"]',
                 'col_ref -> ["airport", ".", "country_name"]',
                 'col_ref -> ["airport", ".", "minimum_connect_time"]',
                 'col_ref -> ["airport", ".", "state_code"]',
                 'col_ref -> ["airport", ".", "time_zone_code"]',
                 'col_ref -> ["airport_service", ".", "airport_code"]',
                 'col_ref -> ["airport_service", ".", "city_code"]',
                 'col_ref -> ["airport_service", ".", "direction"]',
                 'col_ref -> ["airport_service", ".", "miles_distant"]',
                 'col_ref -> ["airport_service", ".", "minutes_distant"]',
                 'col_ref -> ["city", ".", "city_code"]',
                 'col_ref -> ["city", ".", "city_name"]',
                 'col_ref -> ["city", ".", "country_name"]',
                 'col_ref -> ["city", ".", "state_code"]',
                 'col_ref -> ["city", ".", "time_zone_code"]',
                 'col_ref -> ["class_of_service", ".", "booking_class"]',
                 'col_ref -> ["class_of_service", ".", "class_description"]',
                 'col_ref -> ["class_of_service", ".", "rank"]',
                 'col_ref -> ["date_day", ".", "day_name"]',
                 'col_ref -> ["days", ".", "day_name"]',
                 'col_ref -> ["days", ".", "days_code"]',
                 'col_ref -> ["equipment_sequence", ".", "aircraft_code"]',
                 'col_ref -> ["equipment_sequence", ".", "aircraft_code_sequence"]',
                 'col_ref -> ["fare", ".", "fare_airline"]',
                 'col_ref -> ["fare", ".", "fare_basis_code"]',
                 'col_ref -> ["fare", ".", "fare_id"]',
                 'col_ref -> ["fare", ".", "from_airport"]',
                 'col_ref -> ["fare", ".", "one_direction_cost"]',
                 'col_ref -> ["fare", ".", "restriction_code"]',
                 'col_ref -> ["fare", ".", "round_trip_cost"]',
                 'col_ref -> ["fare", ".", "round_trip_required"]',
                 'col_ref -> ["fare", ".", "to_airport"]',
                 'col_ref -> ["fare_basis", ".", "basis_days"]',
                 'col_ref -> ["fare_basis", ".", "booking_class"]',
                 'col_ref -> ["fare_basis", ".", "class_type"]',
                 'col_ref -> ["fare_basis", ".", "discounted"]',
                 'col_ref -> ["fare_basis", ".", "economy"]',
                 'col_ref -> ["fare_basis", ".", "fare_basis_code"]',
                 'col_ref -> ["fare_basis", ".", "night"]',
                 'col_ref -> ["fare_basis", ".", "premium"]',
                 'col_ref -> ["fare_basis", ".", "season"]',
                 'col_ref -> ["flight", ".", "aircraft_code_sequence"]',
                 'col_ref -> ["flight", ".", "airline_code"]',
                 'col_ref -> ["flight", ".", "airline_flight"]',
                 'col_ref -> ["flight", ".", "arrival_time"]',
                 'col_ref -> ["flight", ".", "connections"]',
                 'col_ref -> ["flight", ".", "departure_time"]',
                 'col_ref -> ["flight", ".", "dual_carrier"]',
                 'col_ref -> ["flight", ".", "flight_days"]',
                 'col_ref -> ["flight", ".", "flight_id"]',
                 'col_ref -> ["flight", ".", "flight_number"]',
                 'col_ref -> ["flight", ".", "from_airport"]',
                 'col_ref -> ["flight", ".", "meal_code"]',
                 'col_ref -> ["flight", ".", "stops"]',
                 'col_ref -> ["flight", ".", "time_elapsed"]',
                 'col_ref -> ["flight", ".", "to_airport"]',
                 'col_ref -> ["flight_fare", ".", "fare_id"]',
                 'col_ref -> ["flight_fare", ".", "flight_id"]',
                 'col_ref -> ["flight_leg", ".", "flight_id"]',
                 'col_ref -> ["flight_leg", ".", "leg_flight"]',
                 'col_ref -> ["flight_leg", ".", "leg_number"]',
                 'col_ref -> ["flight_stop", ".", "arrival_airline"]',
                 'col_ref -> ["flight_stop", ".", "arrival_flight_number"]',
                 'col_ref -> ["flight_stop", ".", "arrival_time"]',
                 'col_ref -> ["flight_stop", ".", "departure_airline"]',
                 'col_ref -> ["flight_stop", ".", "departure_flight_number"]',
                 'col_ref -> ["flight_stop", ".", "departure_time"]',
                 'col_ref -> ["flight_stop", ".", "flight_id"]',
                 'col_ref -> ["flight_stop", ".", "stop_airport"]',
                 'col_ref -> ["flight_stop", ".", "stop_days"]',
                 'col_ref -> ["flight_stop", ".", "stop_number"]',
                 'col_ref -> ["flight_stop", ".", "stop_time"]',
                 'col_ref -> ["food_service", ".", "compartment"]',
                 'col_ref -> ["food_service", ".", "meal_code"]',
                 'col_ref -> ["food_service", ".", "meal_description"]',
                 'col_ref -> ["food_service", ".", "meal_number"]',
                 'col_ref -> ["ground_service", ".", "airport_code"]',
                 'col_ref -> ["ground_service", ".", "city_code"]',
                 'col_ref -> ["ground_service", ".", "ground_fare"]',
                 'col_ref -> ["ground_service", ".", "transport_type"]',
                 'col_ref -> ["month", ".", "month_name"]',
                 'col_ref -> ["month", ".", "month_number"]',
                 'col_ref -> ["restriction", ".", "advance_purchase"]',
                 'col_ref -> ["restriction", ".", "application"]',
                 'col_ref -> ["restriction", ".", "maximum_stay"]',
                 'col_ref -> ["restriction", ".", "minimum_stay"]',
                 'col_ref -> ["restriction", ".", "no_discounts"]',
                 'col_ref -> ["restriction", ".", "restriction_code"]',
                 'col_ref -> ["restriction", ".", "saturday_stay_required"]',
                 'col_ref -> ["restriction", ".", "stopovers"]',
                 'col_ref -> ["state", ".", "country_name"]',
                 'col_ref -> ["state", ".", "state_code"]',
                 'col_ref -> ["state", ".", "state_name"]'}

        assert set(valid_actions['table_name']) == \
                {'table_name -> ["aircraft"]',
                 'table_name -> ["airline"]',
                 'table_name -> ["airport"]',
                 'table_name -> ["airport_service"]',
                 'table_name -> ["city"]',
                 'table_name -> ["class_of_service"]',
                 'table_name -> ["date_day"]',
                 'table_name -> ["days"]',
                 'table_name -> ["equipment_sequence"]',
                 'table_name -> ["fare"]',
                 'table_name -> ["fare_basis"]',
                 'table_name -> ["flight"]',
                 'table_name -> ["flight_fare"]',
                 'table_name -> ["flight_leg"]',
                 'table_name -> ["flight_stop"]',
                 'table_name -> ["food_service"]',
                 'table_name -> ["ground_service"]',
                 'table_name -> ["month"]',
                 'table_name -> ["restriction"]',
                 'table_name -> ["state"]'}

    def test_atis_local_actions(self): # pylint: disable=no-self-use
        # Check if the triggers activate correcty
        world = AtisWorld(["show me the flights from denver at 12 o'clock"],
                          database_directory=str(self.database_directory))

        assert set(world.valid_actions['number']) == \
            {'number -> ["0"]',
             'number -> ["1"]',
             'number -> ["1200"]',
             'number -> ["2400"]'}

        world = AtisWorld(["show me the flights from denver at 12 o'clock",
                           "show me the delta or united flights in afternoon"],
                          database_directory=str(self.database_directory))

        assert set(world.valid_actions['number']) == \
                {'number -> ["0"]',
                 'number -> ["1"]',
                 'number -> ["1200"]',
                 'number -> ["2400"]'}

        world = AtisWorld(["i would like one coach reservation for \
                          may ninth from pittsburgh to atlanta leaving \
                          pittsburgh before 10 o'clock in morning 1991 \
                          august twenty sixth"],
                          database_directory=str(self.database_directory))

        assert set(world.valid_actions['number']) == \
                {'number -> ["0"]',
                 'number -> ["1"]',
                 'number -> ["2200"]',
                 'number -> ["1000"]'}

        assert set(world.valid_actions['time_range_start']) == \
                {'time_range_start -> ["0"]'}
        assert set(world.valid_actions['time_range_end']) == \
                {'time_range_end -> ["1200"]'}

        date_binary_expressions = ['biexpr -> ["date_day", ".", "year", binaryop, "1991"]',
                                   'biexpr -> ["date_day", ".", "month_number", binaryop, "8"]',
                                   'biexpr -> ["date_day", ".", "day_number", binaryop, "26"]',
                                   'biexpr -> ["date_day", ".", "year", binaryop, "1991"]',
                                   'biexpr -> ["date_day", ".", "month_number", binaryop, "5"]',
                                   'biexpr -> ["date_day", ".", "day_number", binaryop, "9"]']
        for date_binary_expression in date_binary_expressions:
            assert date_binary_expression in world.valid_actions['biexpr']

    def test_atis_simple_action_sequence(self): # pylint: disable=no-self-use
        world = AtisWorld([("give me all flights from boston to "
                            "philadelphia next week arriving after lunch")],
                          database_directory=str(self.database_directory))
        action_sequence = world.get_action_sequence(("(SELECT DISTINCT city . city_code , city . city_name "
                                                     "FROM city WHERE ( city.city_name = 'BOSTON' ) );"))
        assert action_sequence == ['statement -> [query, ";"]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", "(", conditions, ")"]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                                   'city_city_name_string -> ["\'BOSTON\'"]',
                                   'binaryop -> ["="]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref, ",", col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_name"]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> ["DISTINCT"]']
        action_sequence = world.get_action_sequence(("( SELECT airport_service . airport_code "
                                                     "FROM airport_service "
                                                     "WHERE airport_service . city_code IN ( "
                                                     "SELECT city . city_code FROM city "
                                                     "WHERE city.city_name = 'BOSTON' ) ) ;"))
        assert action_sequence == ['statement -> [query, ";"]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [in_clause]',
                                   'in_clause -> [col_ref, "IN", query]',
                                   'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                                   'where_clause, ")"]',
                                   'where_clause -> ["WHERE", conditions]',
                                   'conditions -> [condition]',
                                   'condition -> [biexpr]',
                                   'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                                   'city_city_name_string -> ["\'BOSTON\'"]',
                                   'binaryop -> ["="]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["city"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["city", ".", "city_code"]',
                                   'distinct -> [""]',
                                   'col_ref -> ["airport_service", ".", "city_code"]',
                                   'table_refs -> [table_name]',
                                   'table_name -> ["airport_service"]',
                                   'select_results -> [col_refs]',
                                   'col_refs -> [col_ref]',
                                   'col_ref -> ["airport_service", ".", "airport_code"]',
                                   'distinct -> [""]']
        action_sequence = world.get_action_sequence(("( SELECT airport_service . airport_code "
                                                     "FROM airport_service WHERE airport_service . city_code IN "
                                                     "( SELECT city . city_code FROM city "
                                                     "WHERE city.city_name = 'BOSTON' ) AND 1 = 1) ;"))
        assert action_sequence == \
               ['statement -> [query, ";"]',
                'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                'where_clause, ")"]',
                'where_clause -> ["WHERE", conditions]',
                'conditions -> [condition, conj, conditions]',
                'conditions -> [condition]',
                'condition -> [biexpr]',
                'biexpr -> [value, binaryop, value]',
                'value -> [pos_value]',
                'pos_value -> [number]',
                'number -> ["1"]',
                'binaryop -> ["="]',
                'value -> [pos_value]',
                'pos_value -> [number]',
                'number -> ["1"]',
                'conj -> ["AND"]',
                'condition -> [in_clause]',
                'in_clause -> [col_ref, "IN", query]',
                'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                'where_clause, ")"]',
                'where_clause -> ["WHERE", conditions]',
                'conditions -> [condition]',
                'condition -> [biexpr]',
                'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                'city_city_name_string -> ["\'BOSTON\'"]',
                'binaryop -> ["="]',
                'table_refs -> [table_name]',
                'table_name -> ["city"]',
                'select_results -> [col_refs]',
                'col_refs -> [col_ref]',
                'col_ref -> ["city", ".", "city_code"]',
                'distinct -> [""]',
                'col_ref -> ["airport_service", ".", "city_code"]',
                'table_refs -> [table_name]',
                'table_name -> ["airport_service"]',
                'select_results -> [col_refs]',
                'col_refs -> [col_ref]',
                'col_ref -> ["airport_service", ".", "airport_code"]',
                'distinct -> [""]']
        world = AtisWorld([("give me all flights from boston to "
                            "philadelphia next week arriving after lunch")],
                          database_directory=str(self.database_directory))

        action_sequence = world.get_action_sequence(("( SELECT DISTINCT flight.flight_id "
                                                     "FROM flight WHERE "
                                                     "( flight . from_airport IN "
                                                     "( SELECT airport_service . airport_code "
                                                     "FROM airport_service WHERE airport_service . city_code IN "
                                                     "( SELECT city . city_code "
                                                     "FROM city "
                                                     "WHERE city.city_name = 'BOSTON' )))) ;"))
        assert action_sequence == \
            ['statement -> [query, ";"]',
             'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             'where_clause, ")"]',
             'where_clause -> ["WHERE", "(", conditions, ")"]',
             'conditions -> [condition]',
             'condition -> [in_clause]',
             'in_clause -> [col_ref, "IN", query]',
             'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             'where_clause, ")"]',
             'where_clause -> ["WHERE", conditions]',
             'conditions -> [condition]',
             'condition -> [in_clause]',
             'in_clause -> [col_ref, "IN", query]',
             'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
             'where_clause, ")"]',
             'where_clause -> ["WHERE", conditions]',
             'conditions -> [condition]',
             'condition -> [biexpr]',
             'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
             'city_city_name_string -> ["\'BOSTON\'"]',
             'binaryop -> ["="]',
             'table_refs -> [table_name]',
             'table_name -> ["city"]',
             'select_results -> [col_refs]',
             'col_refs -> [col_ref]',
             'col_ref -> ["city", ".", "city_code"]',
             'distinct -> [""]',
             'col_ref -> ["airport_service", ".", "city_code"]',
             'table_refs -> [table_name]',
             'table_name -> ["airport_service"]',
             'select_results -> [col_refs]',
             'col_refs -> [col_ref]',
             'col_ref -> ["airport_service", ".", "airport_code"]',
             'distinct -> [""]',
             'col_ref -> ["flight", ".", "from_airport"]',
             'table_refs -> [table_name]',
             'table_name -> ["flight"]',
             'select_results -> [col_refs]',
             'col_refs -> [col_ref]',
             'col_ref -> ["flight", ".", "flight_id"]',
             'distinct -> ["DISTINCT"]']

    def test_atis_long_action_sequence(self): # pylint: disable=no-self-use
        world = AtisWorld([("what is the earliest flight in morning "
                            "1993 june fourth from boston to pittsburgh")],
                          database_directory=str(self.database_directory))
        action_sequence = world.get_action_sequence("( SELECT DISTINCT flight.flight_id "
                                                    "FROM flight "
                                                    "WHERE ( flight.departure_time = ( "
                                                    "SELECT MIN ( flight.departure_time ) "
                                                    "FROM flight "
                                                    "WHERE ( flight.departure_time BETWEEN 0 AND 1200 AND "
                                                    "( flight . from_airport IN ( "
                                                    "SELECT airport_service . airport_code "
                                                    "FROM airport_service WHERE airport_service . city_code "
                                                    "IN ( "
                                                    "SELECT city . city_code "
                                                    "FROM city WHERE city.city_name = 'BOSTON' )) "
                                                    "AND flight . to_airport IN ( "
                                                    "SELECT airport_service . airport_code "
                                                    "FROM airport_service "
                                                    "WHERE airport_service . city_code IN ( "
                                                    "SELECT city . city_code "
                                                    "FROM city "
                                                    "WHERE city.city_name = 'PITTSBURGH' )) ) ) ) AND "
                                                    "( flight.departure_time BETWEEN 0 AND 1200 AND "
                                                    "( flight . from_airport IN ( "
                                                    "SELECT airport_service . airport_code "
                                                    "FROM airport_service "
                                                    "WHERE airport_service . city_code IN ( "
                                                    "SELECT city . city_code "
                                                    "FROM city WHERE city.city_name = 'BOSTON' )) "
                                                    "AND flight . to_airport IN ( "
                                                    "SELECT airport_service . airport_code "
                                                    "FROM airport_service WHERE airport_service . city_code IN ( "
                                                    "SELECT city . city_code "
                                                    "FROM city "
                                                    "WHERE city.city_name = 'PITTSBURGH' )) ) ) )   ) ;")

        assert action_sequence == \
                ['statement -> [query, ";"]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", "(", conditions, ")"]',
                 'conditions -> [condition, conj, conditions]',
                 'conditions -> ["(", conditions, ")"]',
                 'conditions -> [condition, conj, conditions]',
                 'conditions -> ["(", conditions, ")"]',
                 'conditions -> [condition, conj, conditions]',
                 'conditions -> [condition]',
                 'condition -> [in_clause]',
                 'in_clause -> [col_ref, "IN", query]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", conditions]',
                 'conditions -> [condition]',
                 'condition -> [in_clause]',
                 'in_clause -> [col_ref, "IN", query]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", conditions]',
                 'conditions -> [condition]',
                 'condition -> [biexpr]',
                 'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                 'city_city_name_string -> ["\'PITTSBURGH\'"]',
                 'binaryop -> ["="]',
                 'table_refs -> [table_name]',
                 'table_name -> ["city"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["city", ".", "city_code"]',
                 'distinct -> [""]',
                 'col_ref -> ["airport_service", ".", "city_code"]',
                 'table_refs -> [table_name]',
                 'table_name -> ["airport_service"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["airport_service", ".", "airport_code"]',
                 'distinct -> [""]',
                 'col_ref -> ["flight", ".", "to_airport"]',
                 'conj -> ["AND"]',
                 'condition -> [in_clause]',
                 'in_clause -> [col_ref, "IN", query]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", conditions]',
                 'conditions -> [condition]',
                 'condition -> [in_clause]',
                 'in_clause -> [col_ref, "IN", query]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", conditions]',
                 'conditions -> [condition]',
                 'condition -> [biexpr]',
                 'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                 'city_city_name_string -> ["\'BOSTON\'"]',
                 'binaryop -> ["="]',
                 'table_refs -> [table_name]',
                 'table_name -> ["city"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["city", ".", "city_code"]',
                 'distinct -> [""]',
                 'col_ref -> ["airport_service", ".", "city_code"]',
                 'table_refs -> [table_name]',
                 'table_name -> ["airport_service"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["airport_service", ".", "airport_code"]',
                 'distinct -> [""]',
                 'col_ref -> ["flight", ".", "from_airport"]',
                 'conj -> ["AND"]',
                 'condition -> [ternaryexpr]',
                 'ternaryexpr -> [col_ref, "BETWEEN", time_range_start, "AND", time_range_end]',
                 'time_range_end -> ["1200"]',
                 'time_range_start -> ["0"]',
                 'col_ref -> ["flight", ".", "departure_time"]',
                 'conj -> ["AND"]',
                 'condition -> [biexpr]',
                 'biexpr -> [col_ref, binaryop, value]',
                 'value -> [pos_value]',
                 'pos_value -> [agg_results]',
                 'agg_results -> ["(", "SELECT", distinct, agg, "FROM", table_name, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", "(", conditions, ")"]',
                 'conditions -> [condition, conj, conditions]',
                 'conditions -> ["(", conditions, ")"]',
                 'conditions -> [condition, conj, conditions]',
                 'conditions -> [condition]',
                 'condition -> [in_clause]',
                 'in_clause -> [col_ref, "IN", query]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", conditions]',
                 'conditions -> [condition]',
                 'condition -> [in_clause]',
                 'in_clause -> [col_ref, "IN", query]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", conditions]',
                 'conditions -> [condition]',
                 'condition -> [biexpr]',
                 'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                 'city_city_name_string -> ["\'PITTSBURGH\'"]',
                 'binaryop -> ["="]',
                 'table_refs -> [table_name]',
                 'table_name -> ["city"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["city", ".", "city_code"]',
                 'distinct -> [""]',
                 'col_ref -> ["airport_service", ".", "city_code"]',
                 'table_refs -> [table_name]',
                 'table_name -> ["airport_service"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["airport_service", ".", "airport_code"]',
                 'distinct -> [""]',
                 'col_ref -> ["flight", ".", "to_airport"]',
                 'conj -> ["AND"]',
                 'condition -> [in_clause]',
                 'in_clause -> [col_ref, "IN", query]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", conditions]',
                 'conditions -> [condition]',
                 'condition -> [in_clause]',
                 'in_clause -> [col_ref, "IN", query]',
                 'query -> ["(", "SELECT", distinct, select_results, "FROM", table_refs, '
                 'where_clause, ")"]',
                 'where_clause -> ["WHERE", conditions]',
                 'conditions -> [condition]',
                 'condition -> [biexpr]',
                 'biexpr -> ["city", ".", "city_name", binaryop, city_city_name_string]',
                 'city_city_name_string -> ["\'BOSTON\'"]',
                 'binaryop -> ["="]',
                 'table_refs -> [table_name]',
                 'table_name -> ["city"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["city", ".", "city_code"]',
                 'distinct -> [""]',
                 'col_ref -> ["airport_service", ".", "city_code"]',
                 'table_refs -> [table_name]',
                 'table_name -> ["airport_service"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["airport_service", ".", "airport_code"]',
                 'distinct -> [""]',
                 'col_ref -> ["flight", ".", "from_airport"]',
                 'conj -> ["AND"]',
                 'condition -> [ternaryexpr]',
                 'ternaryexpr -> [col_ref, "BETWEEN", time_range_start, "AND", time_range_end]',
                 'time_range_end -> ["1200"]',
                 'time_range_start -> ["0"]',
                 'col_ref -> ["flight", ".", "departure_time"]',
                 'table_name -> ["flight"]',
                 'agg -> [agg_func, "(", col_ref, ")"]',
                 'col_ref -> ["flight", ".", "departure_time"]',
                 'agg_func -> ["MIN"]',
                 'distinct -> [""]',
                 'binaryop -> ["="]',
                 'col_ref -> ["flight", ".", "departure_time"]',
                 'table_refs -> [table_name]',
                 'table_name -> ["flight"]',
                 'select_results -> [col_refs]',
                 'col_refs -> [col_ref]',
                 'col_ref -> ["flight", ".", "flight_id"]',
                 'distinct -> ["DISTINCT"]']

    def test_atis_from_json(self):
        line = json.loads(self.data[0])
        for utterance_idx in range(len(line['interaction'])):
            world = AtisWorld([interaction['utterance'] for
                               interaction in line['interaction'][:utterance_idx+1]],
                              database_directory=str(self.database_directory))
            action_sequence = world.get_action_sequence(line['interaction'][utterance_idx]['sql'])
            assert action_sequence is not None
