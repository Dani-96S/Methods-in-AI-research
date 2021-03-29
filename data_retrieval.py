import json
import os
import random


def read_file(filename, form):
    """Reads a json file returns transcription data"""
    result = []
    try:
        with open(filename) as json_file:
            data = json.load(json_file)
        for turn in data["turns"]:
            if form == "log":
                result.append(turn["output"]["transcript"])
            else:
                result.append(turn["transcription"])
    except FileNotFoundError:
        return result
    else:
        return result


def transcribe(dirname):
    """Combines transcriptions of system and user into a dialog"""
    log = read_file(dirname + "/log.json", "log")
    label = read_file(dirname + "/label.json", "label")
    transcription = []
    for system, user in zip(log, label):
        transcription.append("system: " + system)
        transcription.append("  user: " + user)
    return transcription


def get_speech_acts(dirname):
    result = []
    try:
        with open(dirname + "/label.json") as label_file:
            data = json.load(label_file)
            for turn in data["turns"]:
                act = turn["semantics"]["cam"].split("(")[0]
                utterance = turn["transcription"]
                result.append(act + " " + utterance)
    except FileNotFoundError:
        return result
    else:
        return result


def get_directories(root):
    """Returns a list of directories containing log an label files"""
    data_directory = root
    directories = []
    for subdir, _, _ in os.walk(data_directory):
        if "voip" in subdir:
            directories.append(subdir)
    return directories


def write_speechacts():
    with open("data/speechacts_train.txt", "w") as speech_file:
        directories = get_directories("./data/traindata/")
        for directory in directories:
            acts = get_speech_acts(directory)
            for line in acts:
                speech_file.write(line + "\n")
    with open("data/speechacts_test.txt", "w") as speech_file:
        directories = get_directories("./data/testdata/")
        for directory in directories:
            acts = get_speech_acts(directory)
            for line in acts:
                speech_file.write(line + "\n")


def write_transcriptions():
    dirs = get_directories("./data")
    with open("data/all_dialogs.txt", mode="w") as dialog_file:
        for dir in dirs:
            transcription = transcribe(dir)
            for line in transcription:
                dialog_file.write(line+"\n")
            dialog_file.write("\n")


def return_speechacts():
    """Reads all speechacts and returns 2 arrays with train and test data"""
    with open("data/speechacts_train.txt") as speech_file:
        train_data = [s.replace("\n", "") for s in speech_file.readlines()]
    with open("data/speechacts_test.txt") as speech_file:
        test_data = [s.replace("\n", "") for s in speech_file.readlines()]
    return train_data, test_data


def print_transcriptions():
    """Prints random dialogues from data"""
    dirs = get_directories("./data")
    next_transcription = True
    while next_transcription:
        random_sample = random.randint(0, len(dirs))
        transcription = transcribe(dirs[random_sample])
        for line in transcription:
            print(line)
        print("---")
        user_input = input("[Enter] for next transcription, [q] to quit: ")
        if user_input == "q":
            next_transcription = False


if __name__ == "__main__":
    print_transcriptions()
