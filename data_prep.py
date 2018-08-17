import csv
import os
import sys


def open_text(path):
    text = open(path, 'r').read()
    return text


def convert_to_csv(data_path, csv_path, data_name):
    all_data = [["article", "class"]]
    for subdir, dirs, files in os.walk(data_path):
        for file_ in files:
            each_file = []
            print file_
            file_path = subdir + os.path.sep + file_
            author = ""
            file_text = ""
            if data_name == "ccat":
                author = file_path.split("/")[-2]
            elif data_name == "imdb":
                author = file_.split(".")[0]
            file_text = open_text(file_path)
            each_file.append(file_text)
            each_file.append(author)
            all_data.append(each_file)

        file_name = csv_path
        with open(file_name, 'wb') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL, dialect='excel')
            writer.writerows(all_data)

if __name__ == "__main__":
    data_path = sys.argv[1]
    csv_path = sys.argv[2]
    mode = sys.argv[3]
    convert_to_csv(data_path, csv_path, mode)
