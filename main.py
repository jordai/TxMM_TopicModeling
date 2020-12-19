import os
import json
import shutil
import pandas as pd
from DataLoader import load_data

def main():
    data_path = "data/rivm/"
    tweets = load_data(data_path)
    print(tweets.head)

if __name__ == '__main__':
    main()

