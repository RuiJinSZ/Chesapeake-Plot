"""
Parse observation data
"""

from typing import Dict, Tuple

import csv
import numpy as np

from .utils import MONTH_TO_DAY


class Observable(object):
    def __init__(self, date: str, depth: str, value: str, totaldepth: str) -> None:
        date_detail = date.split('/')
        self.month = int(date_detail[0])
        self.day   = int(date_detail[1])
        self.year  = int(date_detail[2])
        self.depth = float(depth)
        self.value = float(value)

        if totaldepth == "":
            self.totaldepth = 0.0
        else:
            self.totaldepth = float(totaldepth)


# Return dictionary data
# data[station] contains the information of a certain station
# data[station][parameter] contains the information of a certain parameter in this station
def read_csv(observation_csv: str) -> Dict:
    data = {}
    current_station = "null-station"
    current_parameter = "null-parameter"
    with open(observation_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        reader.__next__()
        for row in reader:
            if (len(row) < 1): break
            if (row[0] != current_station):
                current_station = row[0]
                data[current_station] = {}
                current_parameter = "null-parameter"
            if (row[17] != current_parameter):
                current_parameter = row[17]
                data[current_station][current_parameter] = []
            if (row[13] == '' or row[19] == ''): continue
            data[current_station][current_parameter].append(Observable(row[8], row[13], row[19], row[10]))
    return data


def parse_observation(
    observation_csv: str,
    year: int,
    station: str,
    species: str,
    molecular_weight: float,
) -> Tuple[np.ndarray]:
    data = read_csv(observation_csv)

    month = []
    day   = []
    depth = []
    value = []
    totaldepth = []
    for record in data[station][species]:
        if record.year == year:
            month.append(record.month)
            day.append(record.day)
            depth.append(record.depth)
            value.append(record.value)
            totaldepth.append(record.totaldepth)
    month = np.array(month)
    day = np.array(day)
    depth = np.array(depth)
    value = np.array(value)
    totaldepth =  np.array(totaldepth)

    # Observed data use month-day to mark time, but day is the more suitable unit for computation
    time = np.empty(month.shape)
    for i in range(time.shape[0]):
        time[i] = MONTH_TO_DAY[month[i]] + day[i]

    # Observed data is in milligram, but mole is the more suitable unit for computation
    value = value * 1000.0 / molecular_weight

    # The representative total depth in Chesapeake bay is 20 meters,
    # so we rescale all depth as if the total depth is the representative value everywhere
    depth = 20 * depth / totaldepth

    return time, depth, value
