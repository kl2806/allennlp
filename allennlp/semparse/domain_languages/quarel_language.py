"""
This module defines a domain language for the QuaRel dataset, a simple domain theory for reasoning
about qualitative relations.
"""
from typing import Callable, List, Dict

from allennlp.semparse.domain_languages.domain_language import DomainLanguage, predicate
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph

class World:
    def __init__(self, number: int) -> None:
        self.number = number


class Direction:
    def __init__(self, number: int) -> None:
        self.number = number


class PropertyString:
    def __init__(self,
                 text: str) -> None:
        self.text = text


class PropertyType:
    def __init__(self,
                text: str) -> None:
        self.text = text


class PropertyValue:
    def __init__(self,
                 quarel_property: PropertyString,
                 direction: Direction,
                 world: World) -> None:
        self.quarel_property = quarel_property
        self.direction = direction
        self.world = world


def make_property_predicate(property_name: str) -> Callable[[Direction, World], PropertyString]:
    def property_function(direction: Direction, world: World) -> PropertyString:
        return PropertyString(Property(property_name), direction, world)
    return property_function


class QuaRelLanguage(DomainLanguage):
    """
    Domain language for the QuaRel dataset.
    """
    def __init__(self,
                 table_graph: KnowledgeGraph = None,
                 theories: List[Dict[str, int]] = [],
                 property_strings: List[str] = None):
        self.table_graph = table_graph

        if property_strings:
            property_string_constants= {f'"{property_string}"' : PropertyString(f'"{property_string}"') for property_string in property_strings}
        else:
            property_string_constants= {}

        constants = {'world1': World(1),
                     'world2': World(2),
                     'higher': Direction(1),
                     'lower': Direction(-1),
                     'high': Direction(1),
                     'low': Direction(-1)}
 
        super().__init__(start_types={int}, allowed_constants={**property_string_constants, **constants})
         
        self.theories = theories
        ''' 
        else:
            self.theories = [{"friction": 1, "speed": -1, "smoothness": -1, "distance": -1, "heat": 1},
                             {"speed": 1, "time": -1},
                             {"speed": 1, "distance": 1},
                             {"time": 1, "distance": 1},
                             {"weight": 1, "acceleration": -1},
                             {"strength": 1, "distance": 1},
                             {"strength": 1, "thickness": 1},
                             {"mass": 1, "gravity": 1},
                             {"flexibility": 1, "breakability": -1},
                             {"distance": 1, "loudness": -1, "brightness": -1, "apparentSize": -1},
                             {"exerciseIntensity": 1, "amountSweat": 1}]
 
        for quarel_property in ["friction", "speed", "distance", "heat", "smoothness", "acceleration",
                                "amountSweat", "apparentSize", "breakability", "brightness", "exerciseIntensity",
                                "flexibility", "gravity", "loudness", "mass", "strength", "thickness",
                                "time", "weight"]:
            func = make_property_predicate(quarel_property)
            self.add_predicate(quarel_property, func)
        '''

        # ``and`` is a reserved word, so we add it as a predicate here instead of using the decorator.
        def and_function(quarel_0: PropertyString, quarel_1: PropertyString) -> PropertyString:
            # If the two relations are compatible, then we can return either of them.
            if self._check_quarels_compatible(quarel_0, quarel_1):
                return quarel_0
            else:
                return None
        self.add_predicate('and', and_function)

    @predicate
    def property_value(self,
                      property_string: PropertyType,
                      direction: Direction,
                      world: World) -> PropertyValue:
        return PropertyValue(property_string, direction, world)
    
    @predicate
    def define_and_infer(self, quarel_definition: int, answer: int) -> int:
        return answer

    @predicate
    def define_positive_quarel(self, quarel_0: PropertyString, quarel_1: PropertyString) -> int:
        self.add_constant(quarel_0.text, PropertyType(quarel_0.text))
        self.add_constant(quarel_1.text, PropertyType(quarel_1.text))
        self.theories.append({quarel_0.text: 1, quarel_1.text: 1})
        return 1

    @predicate
    def define_negative_quarel(self, quarel_0: PropertyString, quarel_1: PropertyString) -> int:
        self.add_constant(quarel_0.text, PropertyType(quarel_0.text))
        self.add_constant(quarel_1.text, PropertyType(quarel_1.text))
        self.theories.append({quarel_0.text: 1, quarel_1.text: -1})
        return -1

    @predicate
    def infer(self, setup: PropertyValue, answer_0: PropertyValue, answer_1: PropertyValue) -> int:
        """
        Take the question and check if it is compatible with either of the answer choices.
        """
        if self._check_quarels_compatible(setup, answer_0):
            if self._check_quarels_compatible(setup, answer_1):
                # Found two answers
                return -2
            else:
                return 0
        elif self._check_quarels_compatible(setup, answer_1):
            return 1
        else:
            return -1

    def _check_quarels_compatible(self, quarel_0: PropertyType, quarel_1: PropertyType) -> bool:
        if not (quarel_0 and quarel_1):
            return False
        for theory in self.theories:
            if quarel_0.quarel_property.text in theory and quarel_1.quarel_property.text in theory:
                world_same = 1 if quarel_0.world.number == quarel_1.world.number else -1
                direction_same = 1 if quarel_0.direction.number == quarel_1.direction.number else -1
                is_compatible = theory[quarel_0.quarel_property.text] * theory[quarel_1.quarel_property.text] \
                        * world_same * direction_same
                if is_compatible == 1: # pylint: disable=simplifiable-if-statement
                    return True
                else:
                    return False
        return False

    def is_table_entity(self, right_hand_side) -> bool:
        return False

    def __repr__(self):
        return str(self.theories) + str(self.__dict__)
