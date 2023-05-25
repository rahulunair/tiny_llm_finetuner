import sys
import json

import requests


def get_book_from_gutenberg(url):
    response = requests.get(url)
    text = response.text
    chunk_size = 512
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    data = [{"text": chunk} for chunk in chunks]
    with open("book_data.json", "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"  # pride and prejudice
    if len(sys.argv) > 1:
        genre = sys.argv[1].lower()
        if genre == "sci-fi":
            url = None  # todo
        elif genre == "fantasy":
            url = None  # todo
        elif genre == "mystery":
            url = None  # todo
    get_book_from_gutenberg(url)
