import json
from pathlib import Path
from collections import namedtuple
import re
import numpy as np

Features = namedtuple("Features", [
                      "total_time", "avg_event_duration",
                      "n_changes", "n_remove", "n_add", "add_sizes_hist",
                      "testings", "builds", "builds_succeeded"
                      ])


def get_avg_event_duration(assignment):
    created_at = list(
        filter(lambda event: event["text"] == "created", assignment["events"]))
    if len(created_at) != 0:
        created_at = created_at[0]["time"]
        return np.average([event["time"] - created_at for event in assignment["events"]])
    else:
        return None


def add_sizes(assignment):
    all = [event["diff"] for event in assignment["events"]
           if event["text"] == "modified" and type(event["diff"]) == dict]
    add_sizes = []
    for event in all:
        if "add_lines" in event:
            add_sizes.append(len(event["add_lines"]))
    hist, _ = np.histogram(add_sizes, range=(1, 100), bins=10)
    hist[-1] += np.sum(np.array(add_sizes) > 100)
    return hist


def diffs(assignment):
    all = [event["diff"].keys() for event in assignment["events"]
           if event["text"] == "modified" and type(event["diff"]) == dict]
    n_changes = len([k for k in all if "change" in k])
    n_remove = len([k for k in all if "remove_lines" in k])
    n_add = len([k for k in all if "add_lines" in k])
    return n_changes, n_remove, n_add


def features(assignment):
    total_time = assignment["total_time"]
    avg_event_duration = get_avg_event_duration(assignment)
    n_changes, n_remove, n_add = diffs(assignment)
    return Features(total_time=total_time,
                    avg_event_duration=avg_event_duration,
                    n_changes=n_changes,
                    n_remove=n_remove,
                    n_add=n_add,
                    add_sizes_hist=add_sizes(assignment),
                    testings=assignment["testings"],
                    builds=assignment["builds"],
                    builds_succeeded=assignment["builds_succeeded"],
                    )


def find_assignment(assignment_id, data):
    match = re.match(
        r"([A-Za-z])[^/]*/[^/0-9]*(\d+)/[^/0-9]*(\d+).*", assignment_id)
    if match:
        semester, project, task = match.groups()
    else:
        return None
    keys = [k for k in data.keys() if re.match(
        rf"{semester}[^/]*/[^/0-0]*{project}/[^/0-9]*{task}.*c(?:pp)?$", k)]
    if len(keys) == 0:
        return None
    else:
        return data[keys[0]]


def get_student_features(assignment_id, student_id):
    try:
        with open(Path("plagiarism-dataset/stats/A2017") / (student_id + ".json")) as inf:
            data = json.load(inf)
            assignment = find_assignment(assignment_id, data)
            if assignment != None:
                return features(assignment)
    except Exception:
        return None


def flatten(lst):
    res = []
    for v in lst:
        if hasattr(v, "__iter__"):
            res.extend(flatten(v))
        else:
            res.append(v)
    return res


def assignmentToFeatureVector(assignment):
    return np.array(flatten(assignment))
