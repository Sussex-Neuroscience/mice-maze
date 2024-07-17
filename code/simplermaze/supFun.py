import time
import os
from tkinter import Tk
from tkinter import filedialog as fd
import csv
from re import Pattern
from typing import Optional, Callable, Dict, Tuple, List
import re
from datetime import datetime

import pandas as pd
import cv2
import numpy as np


def safe_input(
    prompt: str,
    accepted_pattern: Optional[Pattern] = None,
    check_function: Optional[Callable[[str], bool]] = None,
) -> str:
    """
    Prompt the user for input and validate it against a pattern and/or a custom function.

    This function will repeatedly prompt the user for input until the input satisfies the
    specified validation criteria. The criteria can include a regular expression pattern
    and/or a custom validation function.

    Parameters:
    prompt (str): The message displayed to the user when asking for input.
    accepted_pattern (Optional[str]): A regex pattern that the input must match. If None, no pattern matching is performed.
    check_function (Optional[Callable[[str], bool]]): A function that takes the user input as an argument and returns True if the input is valid, False otherwise. If None, no additional validation is performed.

    Returns:
    str: The validated user input.

    Example:
    >>> import re
    >>> def is_positive_number(s):
    ...     return s.isdigit() and int(s) > 0
    >>> safe_input("Enter a positive number: ", accepted_pattern=re.compile(r'^\d+$'), check_function=is_positive_number)
    Enter a positive number: -5
    Invalid input, please try again
    Enter a positive number: 10
    '10'
    """
    user_input = input(prompt)

    matches_pattern = (
        accepted_pattern.match(user_input) if accepted_pattern is not None else True
    )
    passes_check = check_function(user_input) if check_function is not None else True

    while not (matches_pattern and passes_check):
        print("Invalid input, please try again")
        user_input = input(prompt)

        matches_pattern = (
            accepted_pattern.match(user_input) if accepted_pattern is not None else True
        )
        passes_check = (
            check_function(user_input) if check_function is not None else True
        )

    return user_input


def is_valid_date(date_str: str) -> bool:
    """
    Check if the given date string is in the format dd/mm/yyyy and is a valid date.

    Parameters:
    date_str (str): The date string to validate.

    Returns:
    bool: True if the date is valid, False otherwise.
    """
    try:
        datetime.strptime(date_str, "%d/%m/%Y")
        return True
    except ValueError:
        return False


def is_valid_int(s: str) -> bool:
    """
    Check if the given string can be converted to a valid integer.

    This function attempts to convert the input string to an integer. If the conversion
    is successful, it returns True. If a ValueError is raised during the conversion,
    it returns False.

    Parameters:
    s (str): The string to check.

    Returns:
    bool: True if the string can be converted to an integer, False otherwise.

    Example:
    >>> is_valid_int("123")
    True
    >>> is_valid_int("abc")
    False
    >>> is_valid_int("123.45")
    False
    >>> is_valid_int("-456")
    True
    """
    try:
        _ = int(s)
        return True
    except ValueError:
        return False


def collect_metadata(animal_id, session_id):
    """
    Collect metadata for an animal during a session.

    This function prompts the user for various pieces of information about an animal,
    including ear mark identifiers, weight, birth date, and gender. The inputs are
    validated using predefined patterns and functions to ensure correctness.

    Parameters:
    animal_id (str): The unique identifier for the animal.
    session_id (str): The identifier for the session.

    Returns:
    Dict[str, str]: A dictionary containing the collected metadata with keys:
        - "animal ID": The provided animal_id.
        - "session ID": The provided session_id.
        - "ear mark identifier": 'y' or 'n' indicating the presence of ear marks.
        - "animal weight": The weight of the animal in grams as a string.
        - "animal birth date": The birth date of the animal in 'dd/mm/yyyy' format.
        - "animal gender": 'm' or 'f' indicating the assumed gender of the animal.

    Example usage:
    >>> metadata = collect_metadata("A123", "S456")
    Ear mark identifiers? (y/n):
    y
    Insert animal weight (g):
    250
    Insert animal birth date (dd/mm/yyyy):
    01/01/2020
    Insert animal assumed gender (m/f):
    m
    >>> print(metadata)
    {
        "animal ID": "A123",
        "session ID": "S456",
        "ear mark identifier": "y",
        "animal weight": "250",
        "animal birth date": "01/01/2020",
        "animal gender": "m",
    }
    """
    ear_mark = safe_input(
        "Ear mark identifiers? (y/n): \n",
        accepted_pattern=re.compile(r"y|n|Y|N"),
        check_function=lambda i: i in ["y", "n", "Y", "N"],
    ).lower()
    weight = safe_input(
        "Insert animal weight (g): \n",
        accepted_pattern=re.compile(r"\d+"),
        check_function=lambda w: is_valid_int(w) and int(w) > 0,
    ).lower()
    birth_date = safe_input(
        "Insert animal birth date (dd/mm/yyyy): \n",
        accepted_pattern=re.compile(r"([0-3]?[0-9])\/[0-1]?[0-9]\/[0-9]{4}"),
        check_function=is_valid_date,
    )
    gender = safe_input(
        "Insert animal assumed gender (m/f): \n",
        accepted_pattern=re.compile(r"m|f|M|F"),
        check_function=lambda g: g in ["m", "M", "f", "F"],
    ).lower()

    data = {
        "animal ID": animal_id,
        "session ID": session_id,
        "ear mark identifier": ear_mark,
        "animal weight": weight,
        "animal birth date": birth_date,
        "animal gender": gender,
    }

    return data


def save_metadata_to_csv(data: Dict[str, str], new_dir_path: str, file_name: str):
    """
    Save metadata to a CSV file.

    This function takes a dictionary of metadata, converts it into a pandas DataFrame,
    and saves it as a CSV file in the specified directory with the given file name.

    Parameters:
    data (Dict[str, str]): The metadata to save, where keys are column names and values
                           are the corresponding data.
    new_dir_path (str): The directory path where the CSV file will be saved.
    file_name (str): The name of the CSV file.

    Returns:
    None

    Example usage:
    >>> metadata = {
    ...     "animal ID": "A123",
    ...     "session ID": "S456",
    ...     "ear mark identifier": "y",
    ...     "animal weight": "250",
    ...     "animal birth date": "01/01/2020",
    ...     "animal gender": "m",
    ... }
    >>> save_metadata_to_csv(metadata, "/path/to/directory", "metadata.csv")
    Metadata saved to: /path/to/directory/metadata.csv
    """
    df = pd.DataFrame([data])
    csv_path = os.path.join(new_dir_path, file_name)
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to: {csv_path}")


def setup_directories(base_path: str, date_time: str, animal_id: str, session_id: int):
    """
    Set up directories for storing session data.

    This function creates a new directory path based on the provided base path, date and time,
    animal ID, and session ID. It ensures that the directory exists and returns the full path
    to the newly created directory.

    Parameters:
    base_path (str): The base directory where the new session directory will be created.
    date_time (str): The date and time string to include in the directory name.
    animal_ID (str): The animal ID to include in the directory name.
    session_ID (str): The session ID to include in the directory name.

    Returns:
    str: The full path to the newly created directory.

    Example usage:
    >>> base_path = "/path/to/base"
    >>> date_time = "20240101_120000"
    >>> animal_ID = "A123"
    >>> session_ID = "S456"
    >>> new_dir_path = setup_directories(base_path, date_time, animal_ID, session_ID)
    >>> print(new_dir_path)
    /path/to/base/20240101_120000A123S456
    """
    new_directory = f"{date_time}{animal_id}{session_id}"
    new_dir_path = os.path.join(base_path, new_directory)
    ensure_directory_exists(new_dir_path)
    return new_dir_path


def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists at the specified path.

    This function checks if a directory exists at the given path. If the directory does
    not exist, it creates the directory. If the directory already exists, it does nothing.

    Parameters:
    path (str): The path to the directory to check or create.

    Returns:
    None

    Example usage:
    >>> ensure_directory_exists("/path/to/directory")
    Directory created: /path/to/directory
    >>> ensure_directory_exists("/path/to/directory")
    Directory already exists: /path/to/directory
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")


def get_user_inputs() -> Tuple[str, str, int]:
    """
    Prompt the user for animal ID, session ID, and experiment phase.

    This function collects user inputs for animal ID, session ID, and experiment phase.
    It ensures that the inputs are correctly formatted and returns them as a tuple.

    Returns:
    Tuple[str, str, int]: A tuple containing:
        - animal ID (str): The animal ID input by the user (converted to uppercase).
        - session ID (str): The session ID input by the user (converted to lowercase).
        - experiment phase (int): The experiment phase input by the user (as an integer).

    Example usage:
    >>> animal_ID, session_ID, experiment_phase = get_user_inputs()
    Insert animal ID:
    A123
    Insert Session ID:
    s456
    Input experiment phase (1-4):
    2
    >>> print(animal_ID, session_ID, experiment_phase)
    A123 s456 2
    """
    animal_id = safe_input("Insert animal ID: \n").upper()
    session_id = safe_input("Insert Session ID: \n").lower()
    experiment_phase = int(
        safe_input(
            "Input experiment phase (1-4): \n",
            accepted_pattern=re.compile(r"[1-4]"),
        )
    )
    return animal_id, session_id, experiment_phase


def get_current_time_formatted() -> str:
    """
    Returns the current date and time in the ISO-8061 format
    """
    return time.strftime("%Y-%m-%dT%H_%M_%S", time.localtime())


def write_text(text: str = "string", window: Optional[str] = None) -> None:
    """
    Write text onto an image or window using OpenCV's putText function.

    This function adds text onto an image or window specified by `window` parameter,
    using OpenCV's putText function. If `window` is not provided (or is None), the
    text is not displayed.

    Parameters:
    text (str, optional): The text to write on the image. Default is "string".
    window (str, optional): The name of the window or image variable where the text
                            will be drawn. If None, the text is not displayed.

    Returns:
    None

    Example usage:
    >>> write_text("Hello, World!", "myWindow")
    # Draws "Hello, World!" on the window named "myWindow"
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_of_text = (10, 500)
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    if window is not None:
        cv2.putText(
            window,
            text,
            bottom_left_of_text,
            font,
            font_scale,
            font_color,
            thickness,
            line_type,
        )

    return


def start_camera(video_input: int = 0) -> Optional[cv2.VideoCapture]:
    """
    Start capturing video from a camera.

    This function initializes and starts capturing video from a camera specified by
    `videoInput`. If `videoInput` is not provided, it defaults to 0 (the default camera).

    Parameters:
    videoInput (int, optional): Index of the camera to use. Default is 0.

    Returns:
    cv2.VideoCapture or None: A VideoCapture object representing the camera capture,
                              or None if the camera cannot be opened.

    Example usage:
    >>> cap = start_camera()
    >>> if cap:
    >>>     while True:
    >>>         ret, frame = cap.read()
    >>>         cv2.imshow('Camera Feed', frame)
    >>>         if cv2.waitKey(1) & 0xFF == ord('q'):
    >>>             break
    >>>     cap.release()
    >>>     cv2.destroyAllWindows()
    """
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    return cap


def record_video(
    cap: cv2.VideoCapture,
    record_file: str,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> Optional[cv2.VideoWriter]:
    """
    Record video from a camera feed.

    This function initializes a video writer object and starts recording video from
    a camera capture object (`cap`). It writes the recorded video to a file specified
    by `recordFile`.

    Parameters:
    cap (cv2.VideoCapture): The VideoCapture object representing the camera feed.
    recordFile (str): The path to save the recorded video file.
    frame_width (int): Width of the frames in pixels.
    frame_height (int): Height of the frames in pixels.
    fps (float): Frames per second (FPS) of the recorded video.

    Returns:
    cv2.VideoWriter or None: A VideoWriter object for writing the video frames to `recordFile`,
                             or None if there is an issue initializing the VideoWriter.

    Example usage:
    >>> cap = cv2.VideoCapture(0)
    >>> if cap.isOpened():
    >>>     recordFile = 'output.avi'
    >>>     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    >>>     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    >>>     fps = 30.0
    >>>     writer = record_video(cap, recordFile, frame_width, frame_height, fps)
    >>>     while True:
    >>>         ret, frame = cap.read()
    >>>         if not ret:
    >>>             break
    >>>         writer.write(frame)
    >>>         cv2.imshow('Recording', frame)
    >>>         if cv2.waitKey(1) & 0xFF == ord('q'):
    >>>             break
    >>>     writer.release()
    >>>     cap.release()
    >>>     cv2.destroyAllWindows()
    """
    cc = cv2.VideoWriter_fourcc(*"XVID")
    videoFileObject = cv2.VideoWriter(record_file, cc, fps, (frame_width, frame_height))
    return videoFileObject


def grab_n_convert_frame(cameraHandle):
    # NOTE: Unsure if this is still in use, but it seems pretty useless?
    # need elaboration during the meeting

    # capture a frame
    ret, frame = cameraHandle.read()
    # Our operations on the frame come here
    gray = frame
    return gray, ret


def csv_to_dict(file_name: str = "rois.csv") -> Dict[str, dict]:
    """
    Read CSV file into a dictionary format.

    This function reads a CSV file specified by `file_name` into a pandas DataFrame,
    transposes it, and converts it into a dictionary format where each column becomes
    a key and the corresponding row values become the dictionary values.

    Parameters:
    file_name (str, optional): The path to the CSV file. Default is "rois.csv".

    Returns:
    Dict[str, dict]: A dictionary where keys are column names and values are dictionaries
                     containing row data.

    Example usage:
    >>> data_dict = csv_to_dict("data.csv")
    >>> print(data_dict)
    {'column1': {0: 'value1', 1: 'value2', ...}, 'column2': {0: 'value3', 1: 'value4', ...}, ...}
    """
    csv = pd.read_csv(file_name, index_col=0)
    return csv.transpose().to_dict()


def define_rois(
    video_input: int = 0,
    roi_names: List[str] = ["entrance1", "entrance2", "rew1", "rew2", "rew3", "rew4"],
    output_name: str = "rois.csv",
) -> pd.DataFrame:
    """
    Define Regions of Interest (ROIs) and save their coordinates to a CSV file.

    This function initializes a camera capture, displays the video feed, and prompts
    the user to select ROIs for each entry in `roiNames`. It saves the coordinates of
    the selected ROIs to a CSV file specified by `outputName`.

    Parameters:
    videoInput (int, optional): Index of the camera to use. Default is 0.
    roiNames (List[str], optional): List of names for each ROI entry. Default is
                                    ["entrance1", "entrance2", "rew1", "rew2", "rew3", "rew4"].
    outputName (str, optional): Name of the output CSV file to save ROI coordinates.
                                Default is "rois.csv".

    Returns:
    pd.DataFrame: A pandas DataFrame containing the ROI coordinates (xstart, ystart, xlen, ylen).

    Example usage:
    >>> define_rois(videoInput=0, roiNames=["entrance", "exit"], outputName="my_rois.csv")
    please select location of entrance
    please select location of exit
    >>> # Output: Saves ROI coordinates to "my_rois.csv" and returns a DataFrame with ROI data.
    """

    cap = start_camera(video_input)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cap.release()

    rois = {}

    for entry in roi_names:
        print(f"please select location of {entry}")
        rois[entry] = cv2.selectROI("frame", gray)

    df = pd.DataFrame(rois)
    df.index = ["xstart", "ystart", "xlen", "ylen"]
    df.to_csv(output_name, index=["xstart", "ystart", "xlen", "ylen"])
    cv2.destroyAllWindows()

    return df


def grab_cut(
    frame: np.ndarray, xstart: int, ystart: int, xlen: int, ylen: int
) -> np.ndarray:
    """
    Perform GrabCut algorithm on a region of interest (ROI) in a frame.

    This function extracts a region of interest (ROI) from the input `frame` using
    the coordinates specified by `xstart`, `ystart`, `xlen`, and `ylen`. It applies
    the GrabCut algorithm to segment and refine the ROI, returning the segmented region.

    Parameters:
    frame (np.ndarray): The input frame or image where the ROI is extracted from.
    xstart (int): Starting x-coordinate of the ROI in the frame.
    ystart (int): Starting y-coordinate of the ROI in the frame.
    xlen (int): Length (width) of the ROI.
    ylen (int): Height of the ROI.

    Returns:
    np.ndarray: The segmented region of interest (cut) after applying the GrabCut algorithm.

    Example usage:
    >>> cut = grab_cut(frame, xstart=100, ystart=50, xlen=200, ylen=150)
    >>> cv2.imshow('Segmented ROI', cut)
    >>> cv2.waitKey(0)
    >>> cv2.destroyAllWindows()
    """
    cut = frame[ystart : ystart + ylen, xstart : xstart + xlen]
    return cut


def create_trials(
    num_trials: int = 100, experiment_phase: int = 2, nonRepeat: bool = False
) -> pd.DataFrame:
    """
    Create trial sequences based on experiment parameters and conditions.

    This function generates trial sequences based on experiment phase, reward probabilities,
    and trial constraints. It reads grating maps and reward sequences from CSV files, selects
    appropriate stage data based on `experiment_phase`, and constructs trial sequences.

    Parameters:
    num_trials (int, optional): Number of trials to generate. Default is 100.
    experiment_phase (int, optional): Experiment phase/stage (1-4). Default is 2.
    nonRepeat (bool, optional): Flag indicating whether consecutive trials should not reward
                                the same location. Default is False.

    Returns:
    pd.DataFrame: DataFrame containing trial information with columns: ['rewlocation', 'givereward', 'wrongallowed'].

    Example usage:
    >>> trials = create_trials(num_trials=200, experiment_phase=3, nonRepeat=True)
    >>> print(trials.head())
       rewlocation  givereward  wrongallowed
    0           12        True          True
    1           12       False         False
    2           12       False         False
    3           10       False         False
    4           12       False         False
    """

    if not (1 <= experiment_phase <= 4):
        print("Invalid session stage, defaulting to stage 1 - habituation")
        experiment_phase = 1

    grating_map = pd.read_csv("./simplermaze/grating_maps.csv")
    reward_sequences = pd.read_csv("./simplermaze/reward_sequences.csv")
    stage = reward_sequences[reward_sequences.sessionID == f"Stage {experiment_phase}"]
    trials_distribution = dict()

    for index, location in enumerate(stage.rewloc):
        subTrials = int(np.floor(num_trials * list(stage.portprob)[index]))
        probRewardLocation = np.random.choice(
            [True, False],
            num_trials,
            p=[list(stage.rewprob)[index], 1 - list(stage.rewprob)[index]],
        )

        trialTuples = list()

        for i in range(subTrials):
            trialTuples.append(
                (location, probRewardLocation[i], list(stage.wrongallowed)[index])
            )

        trials_distribution[location] = trialTuples

    all_together = list()


    for item in trials_distribution.keys():
        all_together += trials_distribution[item]  # this keeps the list flat
        # (as opposed to a list of lists)

    # now shuffle the list
    np.random.shuffle(all_together)

    if nonRepeat:
        # rearrange the list so that two/three consecutive trials rewarding the same location never happens
        for i in range(4):
            for index in range(len(all_together) - 1):
                if all_together[index][0] == all_together[index + 1][0]:
                    print("repeat coming up... fixing")
                    temp = all_together[index + 1]
                    all_together.append(temp)
                    all_together.pop(index + 1)

    # create DataFrame using data
    trials = pd.DataFrame(
        all_together, columns=["rewlocation", "givereward", "wrongallowed"]
    )

    return trials


def choose_csv() -> str:
    """
    Open a file dialog to choose a CSV file.

    This function opens a file dialog window using Tkinter, allowing the user to select
    a CSV file. Once the file is selected and the dialog is closed, the function returns
    the full path of the selected CSV file.

    Returns:
    str: Full path of the selected CSV file.

    Example usage:
    >>> csv_file = choose_csv()
    >>> print(f"Selected CSV file: {csv_file}")
    Selected CSV file: /path/to/selected/file.csv
    """
    root = Tk()
    # Show the file dialog and get the selected file name
    filename = fd.askopenfilename()
    root.destroy()
    return filename


def empty_frame(
    rows: int = 300,
    roi_names: List[str] = ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"],
):
    """
    Create an empty DataFrame with specified rows and columns.

    This function creates an empty pandas DataFrame with a specified number of `rows`
    and columns representing various measurements and regions of interest (ROIs).

    Parameters:
    rows (int, optional): Number of rows (entries) in the DataFrame. Default is 300.
    roi_names (List[str], optional): List of names for each ROI. Default is
                                     ["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"].

    Returns:
    pd.DataFrame: Empty DataFrame with NaN values and specified columns.

    Example usage:
    >>> df = empty_frame(rows=500, roi_names=["entrance", "exit", "rewardA", "rewardB"])
    >>> print(df.head())
       hit  miss  incorrect  rew_location  area_rewarded  time_to_reward  ... rewardA  rewardB
    0  NaN   NaN        NaN           NaN           NaN             NaN  ...      NaN      NaN
    1  NaN   NaN        NaN           NaN           NaN             NaN  ...      NaN      NaN
    2  NaN   NaN        NaN           NaN           NaN             NaN  ...      NaN      NaN
    3  NaN   NaN        NaN           NaN           NaN             NaN  ...      NaN      NaN
    4  NaN   NaN        NaN           NaN           NaN             NaN  ...      NaN      NaN

    [5 rows x 14 columns]
    """

    columns = [
        "hit",
        "miss",
        "incorrect",
        "rew_location",
        "area_rewarded",
        "time_to_reward",
        "trial_start_time",
        "end_trial_time",
        "mouse_enter_time",
        "first_reward_area_visited",
    ] + roi_names
    data = pd.DataFrame(None, index=range(rows), columns=columns)
    return data


def time_in_millis():
    """
    Get the current time in milliseconds.

    This function retrieves the current time in milliseconds since the epoch
    using the time.time() function and rounds it to the nearest millisecond.

    Returns:
    int: Current time in milliseconds.

    Example usage:
    >>> timestamp = time_in_millis()
    >>> print(f"Current time in milliseconds: {timestamp}")
    Current time in milliseconds: 1648329175000
    """
    millis = round(time.time() * 1000)
    return millis


def write_data(
    file_name: str = "tests.csv",
    mode: str = "a",
    data: List[str] = ["test", "test", "test"],
):
    """
    Write data to a CSV file.

    This function opens or creates a CSV file specified by `file_name` and writes
    the list of strings `data` as a new row using the provided `mode`.

    Parameters:
    file_name (str, optional): Name of the CSV file to write data to. Default is "tests.csv".
    mode (str, optional): File opening mode ('a' for append, 'w' for write). Default is "a".
    data (List[str], optional): List of strings to write as a new row in the CSV file. Default is ["test", "test", "test"].

    Returns:
    None

    Example usage:
    >>> write_data("output.csv", "w", ["data1", "data2", "data3"])
    """
    with open(file_name, mode) as data_file:
        data_writer = csv.writer(data_file, delimiter=",")
        data_writer.writerow(data)