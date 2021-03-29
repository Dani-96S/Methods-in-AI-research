import copy
import string
import json
from utils import norm_input, closest_word, c, all_inputs
from Levenshtein import ratio
from itertools import combinations


def get_word_data():
    """Gets all 'inform' vocabulary with associative types"""
    with open("./data/inform_vocabulary_edited.txt", "r") as file:
        x = []
        for line in file.readlines():
            y = [value for value in line.split()]
            x.append(y)
    return x


def map_word_to_type(test_input, threshold=0.8):
    """maps all data of an input sentence to their types"""
    user_input = test_input.lower()
    for punct in string.punctuation:
        user_input = user_input.replace(punct, "")
    word_index = get_word_data()
    words = []
    types = []
    for user_word in user_input.split(" "):
        wordfound = False
        for word in word_index:
            best_match = 100
            best_word = ()
            if word[0] == user_word:
                wordfound = True
                words.append(user_word)
                types.append("|".join(word[1:]))
            else:
                match = ratio(word[0], user_word)
                if match < best_match:
                    best_match = match
                    best_word = word
        if not wordfound:
            if best_match > threshold:
                words.append(user_word)
                types.append(best_word[1])
            else:
                words.append(user_word)
                types.append("none")
    return list(zip(words, types))


def variable_val_keys():
    with open("data/variable_keywords.json") as json_file:
        var_keys = json.load(json_file)
    with open("data/variable_values.json") as json_file:
        var_val = json.load(json_file)
    return var_val["informable"], var_keys


def deduct_sentence(sentence):
    """Helper function to deduct"""
    s_data = map_word_to_type(sentence)
    all_trees = deduct((s_data, (None, None, None)), tree=[])
    trees, best_tree = tree_filter(all_trees)
    # TODO: better tree selection
    return best_tree


def tree_lose_rapp(tree):
    return_tree = []
    for step, rapp in tree:
        return_tree.append(step)
    return return_tree


def tree_filter(all_trees):
    best_tree = []
    best_tree_score = 100000
    best_trees = []
    best_score = 100
    for treep in all_trees:
        final, finalrapp = treep[-1]
        if len(final) < best_score:
            best_trees = [treep]
            best_score = len(final)
        elif len(final) == best_score:
            best_trees.append(treep)
    for treep in best_trees:
        score = score_tree(treep)
        if score < best_tree_score:
            best_tree_score = score
            best_tree = treep
    return best_trees, best_tree


def score_tree(treep):
    score = 0
    for tree, rapp in treep:
        l, d, r = rapp
        if d:
            score += len(l[0].split(" ")) + len(l[0].split(" "))
    return score


def deduct(wordp, tree=[]):
    """
    Recursively combine the words according to the corresponding rules
    """
    words, rapp = wordp
    derivables = []
    for index, word in enumerate(words):
        if deductable(words, index):
            derivables.append((word, index))
    tree.append(wordp)
    if not derivables:
        return [tree]
    else:
        subtrees = []
        for derivable in derivables:
            wordpair, index = derivable
            word, types = wordpair
            for typ in types.split("|"):
                sen = copy.copy(words)
                sen[index] = (word, typ)
                subsentence = combine(sen, index)
                subtree = deduct(subsentence, tree=copy.copy(tree))
                subtrees += subtree
        return subtrees


def deductable(words, i):
    word, word_type = words[i]
    rules = parse_rules(words[i][1])
    for rule in rules:
        left, direction, right = rule
        if direction == "L" and i > 0:
            l_word, l_word_type = words[i - 1]
            for l_type in l_word_type.split("|"):
                if left == l_type:
                    return True
        elif direction == "R" and i < len(words)-1:
            r_word, r_word_type = words[i + 1]
            for r_type in r_word_type.split("|"):
                if right == r_type:
                    return True
    return False


def combine(orgwords, i):
    """
    Performs the application of the deduction rule
    """
    words = copy.copy(orgwords)
    word, word_type = words[i]
    rules = parse_rules(words[i][1])
    for rule in rules:
        left, direction, right = rule
        if direction == "L" and i > 0:
            l_word, l_word_type = words[i - 1]
            for l_type in l_word_type.split("|"):
                if left == l_type:
                    comb_word = l_word + " " + word
                    del words[i]
                    words[i-1] = (comb_word, right)
                    rapp = ((l_word, l_word_type),
                            direction,
                            (word, word_type))
                    return words, rapp
        elif direction == "R" and i < len(words)-1:
            r_word, r_word_type = words[i + 1]
            for r_type in r_word_type.split("|"):
                if right == r_type:
                    comb_word = word + " " + r_word
                    del words[i]
                    words[i] = (comb_word, left)
                    rapp = ((word, word_type),
                            direction,
                            (r_word, r_word_type))
                    return words, rapp
    return words, (None, None, None)


def parse_rules(rule):
    """Helper function for parse_rule"""
    rules = []
    for r in rule.split("|"):
        rules.append(parse_rule(r))
    return rules


def parse_rule(rule):
    """
    Reads a rule from string
    Returns the expr left of the direction,
    the direction and
    right of the direction
    """
    left = ""
    right = ""
    direction = ""
    counter = 0
    for char in rule:
        if direction:
            right += char
        elif char == "\\" and counter == 0:
            direction = "L"
        elif char == "/" and counter == 0:
            direction = "R"
        elif char == "(":
            counter += 1
            left += char
        elif char == ")":
            counter -= 1
            left += char
        else:
            left += char
    left = remove_outer_par(left)
    right = remove_outer_par(right)
    return left, direction, right


def remove_outer_par(string):
    """Remove outer parentheses of rule"""
    if len(string) > 0:
        if string[0] == "(":
            return string[1:-1]
    return string


def deduct_preferences(sentence, threshold=0.8):
    """
    Extract preferences from sentence using variable values and keys
    """
    var_val, var_keys = variable_val_keys()
    sentence_tree_rapp = deduct_sentence(sentence)
    sentence_tree = tree_lose_rapp(sentence_tree_rapp)
    sentence = norm_input(sentence).split(" ")

    # Find the keywords that indicate a preference
    to_derive = []
    for variable, keys in var_keys.items():
        for key in keys:
            if key in sentence:
                to_derive.append((variable, key))

    # Find the matching values for the preferences in the lowest subtree
    preferences = []
    for der in to_derive:
        var, key = der
        values = var_val[der[0]]
        for value in values:
            best_match = closest_word(value, sentence)
            if best_match:
                subtree = get_preference(sentence_tree, var, best_match, key)
                if subtree[0]:
                    preferences.append((var, value, key, subtree))

    # Remove overlapping trees
    preferences = disjoint_preferences(preferences)
    preferences = order_preferences(preferences, sentence)
    return preferences, sentence_tree_rapp


def get_preference(tree, variable, value, key):
    """
    Iterate through the trees, return the first subtree
    where the keyword and the value occur together
    """
    for subtree in tree:
        for sub_sent in subtree:
            words, words_type = sub_sent
            if (value in words) and (key in words):
                return sub_sent
    return (None, None)


def disjoint_preferences(preferences):
    """
    When two subtrees are equal with the same preference/value, dismiss one
    When two subtrees are equal with different preference/values, dismiss both
    When one subtree encapsulates another, dismiss the first
    """
    exclude = []
    for pref1, pref2 in combinations(preferences, 2):
        var1, val1, key1, tree1 = pref1
        var2, val2, key2, tree2 = pref2
        if (tree1[0] == tree2[0]):
            if (var1, val1) == (var2, val2):
                exclude.append(pref1)
            else:
                exclude.append(pref1)
                exclude.append(pref2)
        elif (tree1[0] in tree2[0]):
            exclude.append(pref2)
        elif (tree2[0] in tree1[0]):
            exclude.append(pref1)
    return list(set(preferences)-set(exclude))


def order_preferences(preferences, sentence):
    ordered_preferences = []
    ordered_sentence = ""
    for word in sentence:
        ordered_sentence += " {}".format(word)
        for preference in preferences:
            var, val, key, subtree = preference
            if subtree[0] in ordered_sentence and \
               preference not in ordered_preferences:
                ordered_preferences.append(preference)
    return ordered_preferences


def print_preferences(preferences):
    for preference in preferences:
        var, val, key, tree = preference
        print("{} {} {}: {}".format(c.B, var.rjust(10), c.E, val))
    for preference in preferences:
        print(" ")
        var, val, key, tree = preference
        print(" Variable: {}".format(var))
        print("    Value: {}".format(val))
        print("  Keyword: {}".format(key))
        print("  Subtree: {}".format(tree))


def print_tree(tree):
    for subtreep in tree:
        sentence = ""
        subtree, rapp = subtreep
        l, d, r = rapp
        for word, typ in subtree:
            sentence += " | " + word
        if d:
            rappstr = "{} {} {}".format(str(l[1]).rjust(14),
                                        d, str(r[1]).ljust(10))
        else:
            rappstr = " ".rjust(27)
        print("{} {}".format(rappstr, sentence))


def print_all(sentence, tree, preferences):
    print(c.r + "{} Sentence {}".format("="*5, "="*30) + c.E)
    print("> " + c.B + sentence + c.E)
    print("{} Tree {}".format("-"*10, "-"*27))
    print_tree(tree)
    print("{} Preferences {}".format("-"*10, "-"*20))
    print_preferences(preferences)
    print(" ")


def test():
    for i, test in enumerate(all_inputs):
        sen = "{}: {}".format(str(i+1).rjust(2), test)
        pref, sent = deduct_preferences(test)
        print_all(sen, sent, pref)


if __name__ == "__main__":
    while True:
        print(c.B + "Please enter a sentence" + c.E)
        print(c.r + "[t]" + c.E + "to test all assignment sentences")
        print(c.r + "[q]" + c.E + "to quit")
        user_input = input()
        if user_input == "q":
            break
        elif user_input == "t":
            test()
        else:
            pref, sent = deduct_preferences(user_input)
            print_all(user_input, sent, pref)
