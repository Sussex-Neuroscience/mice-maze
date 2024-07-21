from dataclasses import dataclass

@dataclass 
class TrialData: 
    trial_ongoing = True 
    rewarded = False 
    entered_maze = False 
    mistake = False 
    has_left_1 = False 
    has_left_2 = False 
    e2_after_e1 = False 
    e1_after_e2 = False 

    ent1_hist = [False, False]
    ent2_hist = [False, False]
    
    visited_any_rew_area = False 
    first_reward_area = "X" 