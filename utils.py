import string
from Levenshtein import ratio


debug = False


class c:
    b = '\033[34m'
    g = '\033[32m'
    y = '\033[33m'
    r = '\033[31m'
    B = '\033[1m'
    E = '\033[0m'


all_inputs = [
   "I'm looking for world food",
   "I want a restaurant that serves world food",
   "I want a restaurant serving Swedish food",
   "I'm looking for a restaurant in the center",
   "I would like a cheap restaurant in the west part of town",
   "I'm looking for a moderately priced restaurant in the west part of town",
   "I'm looking for a restaurant in any area that serves Tuscan food",
   "Can I have an expensive restaurant",
   "I'm looking for an expensive restaurant "
   "and it should serve international food",
   "I need a Cuban restaurant that is moderately priced",
   "I'm looking for a moderately priced restaurant with Catalan food",
   "What is a cheap restaurant in the south part of town",
   "What about Chinese food",
   "I wanna find a cheap restaurant",
   "I'm looking for Persian food please",
   "Find a Cuban restaurant in the center",
]


def uinput(s=""):
    if s:
        print(s)
    user_input = input()
    user_input = norm_input(user_input)
    return user_input


def norm_input(user_input):
    user_input = user_input.lower()
    for punct in string.punctuation:
        user_input = user_input.replace(punct, "")
    return user_input


def closest_word(word, lexicon, treshold=0.8):
    if word in lexicon:
        return word
    else:
        best_match = ""
        for lex in lexicon:
            dist = ratio(word, str(lex))
            if dist > treshold:
                treshold = dist
                best_match = lex
        if best_match:
            return best_match
        return ""


def dbprint(s):
    if debug:
        print("  {}! debug: {}{}".format(c.r, s, c.E))


def talk(s):
    if s[-1] in string.punctuation:
        dot = ""
    else:
        dot = "."
    print("> {}{}{}{}".format(c.y, s.capitalize(), dot, c.E))
