import os
import itertools
import glob
import numpy as np
from pathlib import Path
from collections import namedtuple

Assignment = namedtuple(
    "Assignment", ["name", "plagiarism_pairs", "plagiarisms", "ids"])

data_dir = Path("plagiarism-dataset")


def get_assignments():
    """
    Create list of assignment, each containing a list of student ids and a
    student-student matrix where 1 indicates a plagiarism occurred between two
    students
    """
    with open(data_dir / "ground-truth-anon.txt") as f:
        assignments = []
        for line in f:
            if line[0] == "-":
                name = line.split()[1]
                students = np.array([fname.split(".")[0]
                                    for fname in os.listdir(data_dir / "src" / name) if len(fname.split(".")[0]) > 0])
                assignment = Assignment(name=name,
                                        ids=students,
                                        plagiarism_pairs=set(),
                                        plagiarisms=set())
                assignments.append(assignment)
            elif len(line[:-1].split(",")) > 1:
                students = line[:-1].split(",")
                for s in map(frozenset, itertools.combinations(students, r=2)):
                    assignments[-1].plagiarism_pairs.add(s)
                for s in students:
                    assignments[-1].plagiarisms.add(s)
            elif len(line[:-1]) > 0:
                assignments[-1].plagiarisms.add(line[:-1])
        return assignments


def get_path(assignment, student_id):
    """Get path for a student's assignment submission"""
    return glob.glob(str(data_dir / "src" / assignment.name / (student_id + ".c*")))[0]


assignments = get_assignments()

# assignment_names = [
#     "A2017/Z1/Z1",
#     "A2017/Z1/Z3",
#     "A2017/Z1/Z4",
#     "A2017/Z2/Z1",
#     "A2017/Z2/Z3",
#     "A2017/Z2/Z4",
#     "A2017/Z3/Z1",
#     "A2017/Z3/Z3",
#     "A2017/Z3/Z4",
#     "A2017/Z4/Z1",
#     "A2017/Z4/Z3",
#     "A2017/Z4/Z4",
#     "A2017/Z5/Z1",
#     "A2017/Z5/Z3",
#     "A2017/Z5/Z4"
#     "A2017/Z1/Z2",
#     "A2017/Z2/Z2",
#     "A2017/Z3/Z2",
#     "A2017/Z4/Z2",
#     "A2017/Z5/Z2",
# ]

training = [a for a in assignments if a.name in [
    "A2017/Z1/Z1",
    "A2017/Z1/Z3",
    "A2017/Z1/Z4",
    "A2017/Z2/Z1",
    "A2017/Z2/Z3",
    "A2017/Z2/Z4",
    "A2017/Z3/Z1",
    "A2017/Z3/Z3",
    "A2017/Z3/Z4",
    "A2017/Z4/Z1",
    "A2017/Z4/Z3",
    "A2017/Z4/Z4",
    "A2017/Z5/Z1",
    "A2017/Z5/Z3",
    "A2017/Z5/Z4"
]]

testing = [a for a in assignments if a.name in [
    "A2017/Z1/Z2",
    "A2017/Z2/Z2",
    "A2017/Z3/Z2",
    "A2017/Z4/Z2",
    "A2017/Z5/Z2",
]]
