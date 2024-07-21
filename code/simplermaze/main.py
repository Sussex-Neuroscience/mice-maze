import pandas as pd 
import cv2 
import numpy as np 
import serial 

from trial_data import TrialData
import supFun as sf
from serial_connection import Serial

import os 
import logging
from typing import Optional, Dict, Tuple, Literal
import time 

def determine_entrance_history(trial_data: TrialData, entrance=Literal['entrance1', 'entrance2']):
    trial_data.ent1_hist.insert(0, mouse_present[entrance])
    trial_data.ent1_hist.pop(-1)

    if not trial_data.ent1_hist[0] and trial_data.ent1_hist[1]:
        trial_data.has_left_1 = True 
        if trial_data.has_left_2: 
            trial_data.e1_after_e2 = True 
            trial_data.e2_after_e1 = False 

def setup_logging(timestamp: Optional[str] = None):
    """
    Set up and configure logging for the Simple Maze application.

    This function sets up logging for the Simple Maze application, creating a logger
    that writes debug and higher level messages to a log file named with the given
    `timestamp` and logs warnings and higher level messages to the console. The logging
    format includes the timestamp, logger name, log level, and message. If logging is
    disabled in the configuration, all logging is disabled.

    Parameters:
    timestamp (str, optional): A timestamp string to be included in the log file name.
                               If not provided, it defaults to the current time formatted
                               by `sf.get_current_time_formatted()`.

    Returns:
    logging.Logger: Configured logger instance for the Simple Maze application.

    Example usage:
    >>> logger = setup_logging()
    >>> logger.debug("This is a debug message.")
    >>> logger.warning("This is a warning message.")
    """

    if timestamp is None: 
        timestamp = sf.get_current_time_formatted()

    logger = logging.getLogger("simple_maze")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f"{sf.config.output_dir}/{timestamp}_simple_maze.log")

    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Disable all logs at level CRITICAL or below, 
    # effectively disables all logging across the application
    if not sf.config.do_logs: 
        logger.warning('logging disabled, make sure `do_logs` is set to true in `config.yaml`')
        logging.disable(logging.CRITICAL)

    return logger 

def move_motors_to_neutral(grating_maps: pd.DataFrame, serial: Serial):
    logger.info("moving all motors to neutral position")

    for motor in grating_maps:
        motor1 = motor[(motor.find(" ") + 1):]
        serial.set_motor_to_neutral(motor1)

    logger.info("done")

def draw_rois():
    if sf.config.draw_rois: 
        logger.info("defining regions of interest")
        sf.define_rois(
            video_input=sf.config.video_location, 
            roi_names=["entrance1", "entrance2", "rewA", "rewB", "rewC", "rewD"],
            output_name=f"{sf.config.output_dir}/rois1.csv"
        )

    logger.info("done")

    return pd.read_csv(sf.config.roi_paths[0], index_col=0)

def calculate_area_and_thresholds(rois: pd.DataFrame, gray: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    areas, thresholds = {}, {}

    for region in rois: 
        areas[region] = sf.grab_cut(
            gray,
            xstart=rois[region]["xstart"],
            ystart=rois[region]["ystart"],
            xlen=rois[region]["xlen"],
            ylen=rois[region]["ylen"],
        )

        thresholds[region] = np.sum(areas[region])
    return areas, thresholds 

def display_trial_profile(trials: pd.DataFrame, trial: int) -> None:
    print()
    print(f"Preparing Maze #{trial + 1}")
    print("Trial profile: ")
    print(f"\tReward Location:\t{trials.rewlocation[trial]}")
    print(f"\tGive Reward:\t\t{trials.givereward[trial]}")
    print(f"\tWrong allowed?\t\t{trials.wrongallowed[trial]}")
    print()

def move_motors_to_cueing_position(grating_maps: pd.DataFrame, experiment_phase: int, trials: pd.DataFrame, trial: int):
    logger.info("moving motors to cueing positions for trial %s", trial)
    for grt_position in grating_maps[grating_maps['rewloc'] == trials.rewlocation[trial]]:
        if experiment_phase != 1: 
            message = f"grt{grt_position}\n"
        else: 
            message = f"grt{grt_position[0:2]} 0\n"
            
        serial.write(message)
    logger.info("done")

def find_mouse(gray: np.ndarray, rois: pd.DataFrame, areas: Dict[str, np.ndarray], thresholds: [str, np.ndarray]) -> Dict[str, bool]:
    logger.info("attempting to find mouse")
    mouse_present = {region: False for region in rois}

    for region in rois:
        areas[region] = sf.grab_cut(
            gray,
            xstart=rois[region]["xstart"],
            ystart=rois[region]["ystart"],
            xlen=rois[region]["xlen"],
            ylen=rois[region]["ylen"],
        )

        mouse_present[region] = np.sum(areas[region]) < thresholds[region] / 2

    logger.info("done")

    return mouse_present


if __name__ == "__main__":
    experiment_start = sf.get_current_time_formatted()

    logger = setup_logging(experiment_start)
    print(f"Logs will be written to {logger.handlers[0].baseFilename}")

    grating_maps = pd.read_csv(sf.config.grating_maps_path)
    logger.info("read grating maps")
    
    sf.ensure_directory_exists(sf.config.output_dir) 

    if sf.config.testing: 
        print("Running in testing mode")
        experiment_phase = 3 
        logger.info("running in testing mode")
        trials = sf.create_trials(
            num_trials=100,
            experiment_phase=experiment_phase, 
            nonRepeat=True,
            grating_map=grating_maps,
        )
        logger.info("created trials")
        trials.to_csv(os.path.join(sf.config.output_dir, f"{experiment_start}_trails_before_session.csv"))
        logger.info("saved trials")

        video_file = os.path.join(sf.config.output_dir, f"{experiment_start}_test.mp4")
        trial_ids = trials 
    else: 
        logger.info("running in experiment mode")
        animal_id, session_id, experiment_phase = sf.get_user_inputs()

        logger.info("setting up directories")
        new_dir_path = sf.setup_directories(
            sf.config.output_dir,
            experiment_start,
            animal_id,
            session_id
        )

        recording_name = f"{experiment_start}_{animal_id}.mp4"
        recording_file = os.path.join(sf.config.output_dir, recording_name)
        logger.info("saving video to %s", recording_file)

        logger.info("collecting metadata")
        metadata = sf.collect_metadata(animal_id, session_id)

        logger.info("saving metadata")
        sf.save_metadata_to_csv(metadata, sf.config.output_dir, f"{experiment_start}_{animal_id}")

        logger.info("creating trials")
        trials = sf.create_trials(num_trials=100, experiment_phase=experiment_phase, grating_map=grating_maps)
        logger.info("saving trials")
        trials.to_csv(os.path.join(sf.config.output_dir, "trials_before_session.csv"))
        print("Please select the file containing trials (default: `trials_before_session.csv`)")
        trial_ids = pd.read_csv(sf.choose_csv())

    serial = Serial()
    rois = draw_rois()

    thresholds = {region: 0 for region in rois}
    areas_rewarded = []
    hits = []
    miss = []
    incorrect = [] 
    
    has_visited = {}
    mouse_present = {}

    cap = sf.start_camera(sf.config.video_location)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if sf.config.record_video:
        logger.info("recording video to %s", recording_file)
        video_file_object = sf.record_video(cap, recording_file, width, height, fps)

    valid, gray = cap.read() 
    ret, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    logger.info("calculating areas and thresholds")


    areas, thresholds = calculate_area_and_thresholds(rois, gray)

    session_start = time.time()

    data = sf.empty_frame(len(trials.index))
    sf.write_data(
        file_name=os.path.join(sf.config.output_dir, f"{experiment_start}_session_data.csv"),
        mode="w",
        data=data.head(0)
    )
    
    cv2.namedWindow("Binary Maze with Regions of Interest", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    
    absolute_start_time = sf.time_in_millis()

    for trial in trials.index: 
        logger.info("setting up trial %d", trial)

        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: 
            logger.info("user interupted trials")
            break

        trial_start = sf.time_in_millis() - absolute_start_time

        if trial == 0: 
            time_old_frame = sf.time_in_millis() - trial_start

        data.loc[trial, "trial_start_time"] = trial_start
        data.loc[trial, "rew_location"] = trials.rewlocation[trial]

        display_trial_profile(trials, trial)

        print("Moving all gratings to neutral positions...")

        move_motors_to_neutral(grating_maps, serial)

        print("Done")

        print("\nMoving motors to cueing positions for this trial")

        move_motors_to_cueing_position(grating_maps, experiment_phase, trials, trial)
            
        print("Done") 

        has_visited = {region: False for region in rois} 

        trial_data = TrialData()

        while trial_data.trial_ongoing: 
            if sf.config.pause_between_frames: 
                time.sleep(0.05)
            
            valid, gray_original = cap.read() 
            ret, gray = cv2.threshold(gray_original, 180, 255, cv2.THRESH_BINARY)

            frame_time = sf.time_in_millis() - absolute_start_time

            if not valid:
                logger.error("unable to read frame, stream ended?")
                break 
                
            if sf.config.record_video:
                video_file_object.write(gray_original)
            
            for region in rois: 
                x, y, w, h = rois[region]

                cv2.rectangle(
                    gray, 
                    (x, y),
                    (x + w, y + h),
                    color=(120, 0, 0),
                    thickness=2
                )
            
            
        cv2.imshow("Binary Maze with Regions of Interest", gray)
        cv2.imshow("Original Image", gray_original)
        
        # if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:  # Quit on 'q' or ESC
        #     logger.info("user interupted trials")
        #     break
