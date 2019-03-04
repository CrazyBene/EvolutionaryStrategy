import csv
import numpy as np

def add_csv_row(path, all_fitness):
    with open(path, mode="a", newline="") as file:
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerow(all_fitness)