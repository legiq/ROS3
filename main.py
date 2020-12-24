from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2

class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']
        # getting the initial picture
        obs, _, _, _ = env.step([0,0])
        # convect in for work with cv
        img = cv2.cvtColor(np.ascontiguousarray(obs), cv2.COLOR_BGR2RGB)
        
        condition = True
        while condition:
        
            obs, reward, done, info = env.step([1, 0])
            img = np.ascontiguousarray(obs)
            
            lower_color_bound = np.array([190,180,0])
            upper_color_bound = np.array([255,255,150])
            
            mask = cv2.inRange(img, lower_color_bound, upper_color_bound)
            
            if (np.sum(mask) > 0.05 * mask.shape[0] * mask.shape[1] * 255):
                obs, reward, done, info = env.step([0, 0])
                condition = False
                
            env.render()