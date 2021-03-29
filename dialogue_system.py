from deduction_algorithm import deduct_preferences, variable_val_keys
from text_classification import SpeechActModel
from utils import uinput, dbprint, talk, closest_word, uinput
import csv
import json


class AnswerTemplates:
    """
    Utterances and templates
    """
    @staticmethod
    def pricerange_str(restaurant):
        if restaurant["pricerange"] == "expensive":
            pricerange = "an {}".format(restaurant["pricerange"])
        if restaurant["pricerange"] == "moderate":
            pricerange = "a moderately priced"
        else:
            pricerange = "a {}".format(restaurant["pricerange"])
        return pricerange

    @staticmethod
    def welcome():
        talk("Hello, welcome to our restaurant system.")
        talk("You can ask for a restaurant by area, price range or food type.")
        print("")
        talk("How may I help you?")

    @staticmethod
    def hello():
        talk("Hi! How can I help?")

    @staticmethod
    def atyourservice():
        talk("Glad to be of service!")

    @staticmethod
    def goodbye():
        talk("Glad to be of service,")
        talk("Good bye!")

    @staticmethod
    def error():
        talk("I'm sorry Dave, I'm afraid I can't do that")

    @staticmethod
    def no_restaurant():
        talk("I'm sorry, "
             "there are no restaurants that satisfy your requirements.")

    @staticmethod
    def ask_preference(type="", repeat=False):
        if repeat:
            talk("Sorry, nothing like that exists in my database.")
            talk("Could you specify something else?")
        elif type == "food":
            talk("What type of food would you prefer?")
        elif type == "pricerange":
            talk("What pricerange are you looking for?")
        elif type == "area":
            talk("In which area are you looking for a restaurant?")
        return uinput()

    def recommend_restaurant(self, restaurant, order=[]):
        restaurant_name = restaurant["restaurantname"]
        food = restaurant["food"]
        area = restaurant["area"]
        pricerange = restaurant["pricerange"]
        prefs = self.ordered_prefs(order, food, area, pricerange)
        recommendation = "{} is a restaurant {} {} {}.".\
            format(restaurant_name, *prefs)
        talk(recommendation)

    @staticmethod
    def ordered_prefs(order, food, area, pricerange):
        price_str = "in the {} price range".format(pricerange)
        food_str = "that serves {} food".format(food)
        area_str = "in the {}".format(area)
        all = ["area", "pricerange", "food"]
        ordered = order + list(set(all) - set(order))
        printorder = []
        for item in ordered:
            if item == "food":
                printorder.append(food_str)
            elif item == "area":
                printorder.append(area_str)
            elif item == "pricerange":
                printorder.append(price_str)
        return printorder

    def request_inform(self, restaurant, req):
        noinfo = []
        requests = []
        for request in req:
            if restaurant[request]:
                requests.append(request)
            else:
                noinfo.append(request)
        inform = ""
        for i, request in enumerate(requests):
            if i == 0:
                rest = "of {}".format(restaurant["restaurantname"])
                conn = ""
            elif i == len(requests)-1:
                rest = ""
                conn = " and "
            else:
                rest = ""
                conn = ", "
            if request == "addr" and "postcode" not in requests:
                inform += "{}the address {} is {}".\
                    format(conn, rest, restaurant["addr"])
            if request == "addr" and "postcode" in requests:
                inform += "{}the address {} is {}, {}".\
                    format(conn, rest, restaurant["addr"],
                           restaurant["postcode"])
            if request == "postcode" and "addr" not in requests:
                inform += "{}the postcode {} is {}".\
                    format(conn, rest, restaurant["postcode"])
            if request == "phone" and rest == "":
                inform += "{}their phone number is {}".\
                    format(conn, restaurant["phone"])
            if request == "phone" and rest != "":
                inform += "{}the phone number {} is {}".\
                    format(conn, rest, restaurant["phone"])
            if request == "food" and rest == "":
                inform += "{}they serve {} food".\
                    format(conn, restaurant["food"])
            if request == "food" and rest != "":
                inform += "{}{} serves {} food".\
                    format(conn, restaurant["restaurantname"],
                           restaurant["food"])
            if request == "pricerange" and rest == "":
                inform += "{}it is {} restaurant".\
                    format(conn, self.pricerange_str(restaurant))
            if request == "pricerange" and rest != "":
                inform += "{}{} is {} restaurant".\
                    format(conn, restaurant["restaurantname"],
                           self.pricerange_str(restaurant))
        talk(inform)
        if noinfo:
            self.inform_noinfo(restaurant["restaurantname"], noinfo)

    @staticmethod
    def inform_noinfo(restaurant, noinfo):
        inform = ""
        req = ""
        for i, request in enumerate(noinfo):
            if request == "phone":
                req = "a phone number"
            elif request == "addr":
                req = "an address"
            elif request == "postcode":
                req = "a postcode"
            elif request == "food":
                req = "information about the kitchen"
            elif request == "pricerange":
                req = "information about the price"
            if i == 0:
                inform += "I don't have {}".format(req)
            elif i == len(noinfo) - 1:
                inform += " or {}".format(req)
            else:
                inform += ", {}".format(req)
        inform += " for {}".format(restaurant)
        talk(inform)


class DialogSystem:
    """
    Main class for dialog system
    """
    def __init__(self, sam=None):
        self.active = True
        # speechact classifier
        if not sam:
            self.sam = SpeechActModel(retrain=True, n_epochs=5, embd_size=512,
                                      lstm_size=64, drop_rate=0.5)
        else:
            self.sam = sam
        # keywords for information requests
        self.reqkeys = self.keywords_from_json()
        # what slots are not filled yet
        self.todo = {"area": False,
                     "food": False,
                     "pricerange": False}
        # the information in the slots
        # if preference[type] == "" and todo[type] == True
        # then the user has no preference
        self.preference = {"area": "",
                           "food": "",
                           "pricerange": ""}
        # template engine
        self.template = AnswerTemplates()
        # the order in which the user initially requested information
        self.order = []
        # restaurant information
        self.restaurant_info = self.restaurants_from_csv()
        # restaurants that fulfill the current requirements
        self.restaurants = []
        # restaurants that have already been recommended
        self.recommended_restaurants = []
        # the current recommendation
        self.current_restaurant = None

    def interact(self):
        """
        The main interaction loop
        """
        self.template.welcome()
        while self.active:
            user_input = uinput()
            speechact = self.sam.sentence_prediction(user_input, echo=False)
            dbprint("speechact: {}".format(speechact))
            self.next_action(user_input=user_input, act=speechact)
        print("exiting...")

    def next_action(self, user_input="", act=None):
        """
        Choosing what to do with a specific speech act
        """
        if act == 'inform' or act == 'reqalts':
            self.examine_input(user_input)
            self.find_restaurants()
            self.fill_restaurant_slots()
            self.recommend_next_restaurant()
        elif act == 'request':
            req = []
            for word in user_input.split(" "):
                for var, keywords in self.reqkeys.items():
                    best_match = closest_word(word, keywords)
                    if best_match and var not in req:
                        req.append(var)
            if req:
                self.template.request_inform(self.current_restaurant, req)
            else:
                self.template.error()
        elif act == 'hello':
            self.template.hello()
        elif act == 'bye':
            self.template.goodbye()
            self.active = False
        elif act == 'thankyou':
            self.template.atyourservice()
        else:
            self.template.error()

    def recommend_next_restaurant(self):
        """
        Recommend a restaurant for the current preferences
        Does not recommend a restaurant that was already recommended
        """
        current = None
        if not self.restaurants:
            self.template.no_restaurant()
        else:
            while not current and self.restaurants:
                if self.restaurants[-1] not in self.recommended_restaurants:
                    current = self.restaurants.pop()
                else:
                    self.restaurants.pop()
            if current:
                self.current_restaurant = current
                self.recommended_restaurants.append(self.current_restaurant)
                self.template.recommend_restaurant(self.current_restaurant,
                                                order=self.order)
            else:
                self.template.no_restaurant()

    def examine_input(self, user_input):
        """
        Examine user_input and change current preferences accordingly
        """
        preferences, tree = deduct_preferences(user_input)
        for p in preferences:
            pref_type = p[0]
            self.todo[pref_type] = True
            if p[1] != "any":
                self.preference[pref_type] = p[1]
            self.order.append(pref_type)

    def parse_input_preference(self, user_input, type=""):
        """
        When the user specifies a preferences,
        parse and return proper preference
        Also check for no-preference
        """
        if self.apathetic_user(user_input):
            return "any"
        values, _ = variable_val_keys()
        if type:
            values = values[type]
        for value in values:
            if value in user_input:
                return value
        speechact = self.sam.sentence_prediction(user_input, echo=False)
        if speechact == "negate":
            return "any"
        elif speechact == "bye":
            self.template.goodbye()
            self.active = False
            return "bye"
        else:
            return ""

    @staticmethod
    def apathetic_user(user_input):
        """
        Return True if user states a lack of preference
        """
        if not user_input:
            return True
        if "any" in user_input:
            return True
        if "dont care" in user_input:
            return True
        return False

    def find_restaurants(self):
        """
        Collect all restaurants that satisfy the current preferences
        """
        dbprint(" area: {}".format(self.preference["area"]))
        dbprint(" food: {}".format(self.preference["food"]))
        dbprint("price: {}".format(self.preference["pricerange"]))
        restaurants = []
        for element in self.restaurant_info:
            area, food, price = False, False, False
            if element["area"] == self.preference["area"]\
               or not self.preference["area"]:
                area = True
            if element["food"] == self.preference["food"]\
               or not self.preference["food"]:
                food = True
            if element["pricerange"] == self.preference["pricerange"]\
               or not self.preference["pricerange"]:
                price = True
            if area and food and price:
                restaurants.append(element)
        self.restaurants = restaurants

    def fill_restaurant_slots(self):
        """
        For all unspecified preferences, ask to specify a preferences
        """
        while not (self.todo["food"] and
                   self.todo["area"] and
                   self.todo["pricerange"]):
            if len(self.restaurants) <= 1:
                # If there is one restaurant, the system should recommend that one,
                # even if not all preferences are specified.
                # If there are no restaurants, the system should say so.
                break
            else:
                for key, value in self.todo.items():
                    if not value:
                        todo_key = key
                        break
                i = 0
                repeat = False
                while not self.todo[todo_key] and i <= 3:
                    # Repeat the question until either the preference is specified
                    # or the system does not understand max. 3 times
                    val = self.template.ask_preference(type=todo_key,
                                                       repeat=repeat)
                    pref = self.parse_input_preference(val, type=todo_key)
                    repeat = False
                    if pref == "any":
                        self.todo[todo_key] = True
                        if todo_key not in self.order:
                            self.order.append(todo_key)
                    elif pref == "bye":
                        break
                    elif i > 2:
                        self.todo[todo_key] = True
                        if todo_key not in self.order:
                            self.order.append(todo_key)
                        i += 1
                    elif not pref:
                        repeat = True
                        i += 1
                    else:
                        self.todo[todo_key] = True
                        self.preference[todo_key] = pref
                        if todo_key not in self.order:
                            self.order.append(todo_key)
            self.find_restaurants()

    @staticmethod
    def restaurants_from_csv():
        """
        Get restaurant info from CSV file
        """
        with open('./data/restaurantinfo.csv', newline='\n') as csvfile:
            reader = csv.DictReader(csvfile)
            restaurants = [row for row in reader]
        return restaurants

    @staticmethod
    def keywords_from_json():
        """
        Get keyword info from Json file
        """
        with open('./data/reqkeywords.json') as json_file:
            keys = json.load(json_file)
        return keys


if __name__ == "__main__":
    print("Training model...")
    sam = SpeechActModel(retrain=True, n_epochs=5, embd_size=256,
                         lstm_size=64, drop_rate=0.5)
    while True:
        ds = DialogSystem(sam=sam)
        ds.interact()
        user = uinput(s="\nPress [Enter] to start over, [q] to quit")
        if user == "q":
            break
