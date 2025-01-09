import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time

### YOUR IMPORTS HERE ###
from queue import PriorityQueue
import heapq
import matplotlib.pyplot as plt

# defines a basic node class
class Node:
    def __init__(self, x_in, y_in, theta_in, g_cost_in, f_cost_in, id_in, parent_node_in):
        self.x = x_in
        self.y = y_in
        self.theta = theta_in
        self.g_cost = g_cost_in
        self.f_cost = f_cost_in
        self.id = id_in
        self.parent_node = parent_node_in

    def printme(self):
        print("\tNode id", self.id,":", "x =", self.x, "y =",self.y, "theta =", self.theta, "parentid:", self.parentid)


class path_planning:
    def __init__(self, start_config, nav_num, h_num, collision_fn):
        self.start_config = start_config
        self.nav_num = nav_num
        self.nodes_expanded = 0

        if nav_num == 1:
            self.goal_config = (0.3, -1.3, -np.pi/2)
            #draw_sphere_marker((0.3, -1.3, 0), 0.05, (0, 0, 1, 1))
        elif nav_num == 2:
            self.goal_config = (0.3, 1.1, 0)
            #draw_sphere_marker((0.3, 1.1, 0), 0.05, (0, 1, 0, 1))
        elif nav_num == 3:
            self.goal_config = (2.6, 1.1, 0)
            #draw_sphere_marker((2.6, 1.1, 0), 0.05, (0, 0, 0, 1))
        else:
            self.goal_config = (2.6, -1.3, -np.pi/2)
            #draw_sphere_marker((2.6, -1.3, 0), 0.05, (1, 0, 0, 1))

        self.h_num = h_num

        self.moves = [(0.1, 0), (0, 0.1), (-0.1, 0), (0, -0.1), (0.1, 0.1), (0.1, -0.1), (-0.1, 0.1), (-0.1, -0.1)]
        self.rots = [-np.pi/2, 0, np.pi/2, np.pi]

        self.collision_fn = collision_fn

    # Calculate g cost
    def g(self, cur_config, new_config):

        g_cost2 = (cur_config[0] - new_config[0])**2 + (cur_config[1] - new_config[1])**2 + min(abs(cur_config[2] - new_config[2]), 2*np.pi - abs(cur_config[2] - new_config[2]))**2 

        return np.sqrt(g_cost2) 

    # Calculate h cost
    def h_euclidean(self, new_config, goal_config):

        h_cost2 = (new_config[0] - goal_config[0])**2 + (new_config[1] - goal_config[1])**2 + min(abs(new_config[2] - goal_config[2]), 2*np.pi - abs(new_config[2] - goal_config[2]))**2 

        return np.sqrt(h_cost2) 

    # Custom heuristic cost
    def h_custom(self, new_config, goal_config):
        
        h_cost2 = np.sqrt((new_config[0] - goal_config[0])**2 + (new_config[1] - goal_config[1])**2) + 1.2*min(abs(new_config[2] - goal_config[2]), 2*np.pi - abs(new_config[2] - goal_config[2]))**2 
        
        return h_cost2


class AStar(path_planning):
    def __init__(self, start_config, nav_num, h_num, collision_fn):
        super().__init__(start_config, nav_num, h_num, collision_fn)
        # Used to track if goal is found
        self.goalFound = False

        # Used to record total cost of the final path;
        self.total_cost = 0
        self.total_path_cost = [100000, 100000]

        # Used to store timestamp
        self.timestamp = [0]

        # used to store final path
        self.path = []

        # Used to store explored collisions
        self.explored_collision = []

        # Used to store explored free space
        self.explored_free = []

        # Push in start node to openList
        self.openList = PriorityQueue()
        self.openSet = set()
        self.closedList = set()

        self.new_id = 0
        self.openList.put((0, self.new_id, Node(round(start_config[0], 2), round(start_config[1],2), start_config[2], 0, 0, self.new_id, None)))     # f_cost, id, Node

    def main_astar(self):
        while not self.openList.empty():
            self.nodes_expanded += 1
            cur_f_cost, cur_id, cur_node = self.openList.get()
            cur_config = (round(cur_node.x, 2), round(cur_node.y, 2), cur_node.theta)
            self.closedList.add(cur_config)

            # Goal found
            if cur_config == self.goal_config:
                self.goalFound = True
                print("Goal found")   
                # Reconstruct path
                while cur_node:
                    cur_config = (round(cur_node.x, 2), round(cur_node.y, 2), cur_node.theta)
                    self.path.append(cur_config)
                    cur_node = cur_node.parent_node
                    if cur_node:
                        parent_config = (round(cur_node.x, 2), round(cur_node.y, 2), cur_node.theta)
                        self.total_cost += super().g(cur_config, parent_config)

                print(f"Path total cost: {round(self.total_cost, 4)}")
                self.total_path_cost.append(self.total_cost)
                self.total_path_cost.append(self.total_cost)
                break

            # Explore neighbors
            for move in self.moves:
                new_x = round(cur_config[0], 2) + round(move[0], 2)
                new_y = round(cur_config[1], 2) + round(move[1], 2)

                for rot in self.rots:
                    new_theta = min(cur_config[2] + rot, 2*np.pi - (cur_config[2] + rot))
                    new_config = (round(new_x, 2), round(new_y, 2), new_theta)

                    # Check if explored
                    if new_config in self.closedList:
                        continue

                    # Check for collisions
                    if self.collision_fn(new_config):
                        continue

                    # Compute costs
                    new_g_cost = self.g(cur_config, new_config) + cur_node.g_cost  # Calculate g cost
                    if self.h_num == 1:
                        new_h_cost = super().h_euclidean(new_config, self.goal_config)  # Calculate h cost
                    else:
                        new_h_cost = super().h_custom(new_config, self.goal_config)

                    new_f_cost = new_g_cost + new_h_cost

                    # Check if no same config exist in open list, if exist, then further check if f_cost is smaller
                    if (new_config not in self.openSet) or not any(new_f_cost >= node[2].f_cost for node in self.openList.queue):
                        # new config not in open list, or f_cost is smaller
                        self.new_id += 1
                        self.openSet.add(new_config)
                        self.openList.put((new_f_cost, self.new_id, Node(new_x, new_y, new_theta, new_g_cost, new_f_cost, self.new_id, cur_node)))

    """Run the A* algorithm"""
    def run(self):
        print("=======================================================================")
        if self.nav_num == 1:
            print("Estimated A* run time: 70s")
        elif self.nav_num == 2:
            print("Estimated A* run time: 100s")
        elif self.nav_num == 3:
            print("Estimated A* run time: 100s")
        else:
            print("Estimated A* run time: 400s")

        if self.h_num == 1:
            print("Running A* with Euclidean heuristic...")
        else:
            print("Running A* with customized heuristic...")
        astar_start_time = time.time()
        self.main_astar()
        astar_end_time = time.time() - astar_start_time
        self.timestamp.append(astar_end_time)
        self.timestamp.append(astar_end_time)
        print(f"A* run time: {round(astar_end_time, 4)}")
        print(f"Total nodes expanded: {self.nodes_expanded}")
        print("Finish running A*")
        print("=======================================================================")

        if(self.goalFound):
            self.path.reverse()
            return self.path
        else:
            print("No Solution Found.\n")



class AnaStar(path_planning):
    def __init__(self, start_config, nav_num, h_num, collision_fn):
        super().__init__(start_config, nav_num, h_num, collision_fn)

        self.path = []
        self.total_path_cost = [100000, 100000]
        self.timestamp = [0]

        self.start_config = start_config

        self.g_cost = {}
        self.e = {}
        self.parent = {}
        self.OPEN = []
        self.closedList = set()
        self.G = 100000
        self.E = np.inf

        self.g_cost[start_config] = 0
        if self.h_num ==1:
            self.e[start_config] = self.G / super().h_euclidean(start_config, self.goal_config)
        else:
            self.e[start_config] = self.G / super().h_custom(start_config, self.goal_config)

        heapq.heappush(self.OPEN, (-self.e[start_config], start_config))

    def improveSolution(self):
        while self.OPEN:
            self.nodes_expanded += 1
            negative_cur_e, cur_config = heapq.heappop(self.OPEN)
            cur_e = -negative_cur_e
            self.closedList.add(cur_config)

            if cur_e < self.E:
                self.E = cur_e

            # Goal found
            if cur_config == self.goal_config:
                
                self.G = self.g_cost[cur_config]

                print(f"G: {round(self.G, 4)}")
                print("Goal found")

                # Reconstruct path
                self.path.clear()
                total_cost = 0
                while cur_config:
                    self.path.append(cur_config)
                    parent_config = self.parent.get(cur_config, None)
                    if parent_config:
                        total_cost += super().g(cur_config, parent_config)
                    cur_config = parent_config

                print(f"Path cost: {round(total_cost, 4)}")
                self.total_path_cost.append(total_cost)
                self.total_path_cost.append(total_cost)

                return

            # Explore neighbors
            for move in self.moves:
                new_x = round(cur_config[0], 2) + round(move[0], 2)
                new_y = round(cur_config[1], 2) + round(move[1], 2)

                for rot in self.rots:
                    new_theta = np.fmod(cur_config[2] + rot + np.pi, 2*np.pi)
                    if new_theta < 0:
                        new_theta += 2*np.pi
                    new_theta -= np.pi

                    new_config = (round(new_x, 2), round(new_y, 2), new_theta)

                    if new_config in self.closedList:
                        continue

                    # Check for collisions
                    if self.collision_fn(new_config):
                        continue
                    
                    # Compute costs
                    new_g_cost = self.g_cost[cur_config] + super().g(cur_config, new_config)  # Calculate g cost
                    if self.h_num == 1:
                        new_h_cost = super().h_euclidean(new_config, self.goal_config)  # Calculate h cost
                    else:
                        new_h_cost = super().h_custom(new_config, self.goal_config)

                    new_f_cost = new_g_cost + new_h_cost

                    if new_config not in self.g_cost or new_g_cost < self.g_cost[new_config]:
                        self.g_cost[new_config] = new_g_cost
                        self.parent[new_config] = cur_config

                        if new_f_cost < self.G:
                            if new_h_cost == 0:
                                self.e[new_config] = 10000000000
                            else:
                                self.e[new_config] = (self.G - self.g_cost[new_config]) / new_h_cost
                            heapq.heappush(self.OPEN, (-self.e[new_config], new_config))

    def update_open_list(self):
        updated_OPEN = []
        while self.OPEN:
            negative_cur_e, cur_config = heapq.heappop(self.OPEN)

            if self.h_num == 1:
                h_cost = super().h_euclidean(cur_config, self.goal_config)
            else:
                h_cost = super().h_custom(cur_config, self.goal_config)
            if h_cost == 0:
                continue

            updated_e = (self.G - self.g_cost[cur_config]) / h_cost

            if self.g_cost[cur_config] + h_cost < self.G:
                heapq.heappush(updated_OPEN, (-updated_e, cur_config))

        self.OPEN = updated_OPEN

    def run(self):
        """Run the ANA* algorithm"""
        print("=======================================================================")
        if self.nav_num == 1:
            print("Estimated ANA* run time: 120s")
        elif self.nav_num == 2:
            print("Estimated ANA* run time: 120s")
        elif self.nav_num == 3:
            print("Estimated ANA* run time: 150s")
        else:
            print("Estimated ANA* run time: 600s")

        if self.h_num == 1:
            print("Running ANA* with Euclidean heuristic...")
        else:
            print("Running ANA* with customized heuristic...")
        print("=======================================================================")
        ana_start_time = time.time()

        ITERATION = 60
        i = 1
        
        while i <= ITERATION and self.OPEN:
            print(f"Running iteration: {i}")
            iter_start_time = time.time()
            self.improveSolution()

            print(f"Current E-suboptimal solution: {round(self.E, 4)}")

            # Update e factors for s in OPEN and prune if g(s) + h(s) >= G
            self.update_open_list()
            self.closedList.clear()

            iter_end_time = time.time()
            self.timestamp.append(iter_end_time - ana_start_time)
            self.timestamp.append(iter_end_time - ana_start_time)
            print("ANA* iteration run time: ", round(iter_end_time - iter_start_time, 4))
            print(f"Total nodes expanded: {self.nodes_expanded}")
            print("=======================================================================")
            i += 1
        print(f"Total nodes expanded: {self.nodes_expanded}")
        print("Finish running ANA*")
        print("=======================================================================")
        self.path.reverse()
        return self.path
        

def main(screenshot=False):

    # initialize PyBullet
    connect(use_gui=True)

    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    # define collision obstacles
    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    
    # Set start config
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))   # (-3.4, 1.4, 0)

    # Draw 4 scenario goal config
    draw_sphere_marker((0.3, -1.3, 0), 0.05, (1, 0, 0, 1))  # nav1
    draw_sphere_marker((0.3, 1.1, 0), 0.05, (0, 1, 0, 1))   # nav2
    draw_sphere_marker((2.6, 1.1, 0), 0.05, (0, 0, 1, 1))   # nav3
    draw_sphere_marker((2.6, -1.3, 0), 0.05, (0, 0, 0, 1))  # nav4

    # Set goal config
    time.sleep(5)
    while True:
        print("=======================================================================")
        print("Specify navigation problem number:")
        print("1: to the left of the wall")
        print("2: right in front of the door")
        print("3: in front of the top right desk")
        print("4: same as hw3")
        nav_num = int(input("Enter navigation problem choice (1~4):"))
        if nav_num in [1, 2, 3, 4]:
            break
        else:
            print("Navigation problem number should be 1~4!")
            print("=======================================================================")

    print("Estimated program toal runtime: 20 mins")

    main_start_time = time.time()
    path_astar = []
    path_anastar = []
    A_star_planner = AStar(start_config, nav_num, 1, collision_fn)          # A* h:euclidean
    path_astar = A_star_planner.run()
    A_star_planner_h2 = AStar(start_config, nav_num, 2, collision_fn)       # A* h:custom
    path_astar_h2 = A_star_planner_h2.run()

    ANA_star_planner = AnaStar(start_config, nav_num, 1, collision_fn)      # ANA* h:euclidean
    path_anastar = ANA_star_planner.run()
    ANA_star_planner_h2 = AnaStar(start_config, nav_num, 2, collision_fn)   # ANA* h:custom
    path_anastar_h2 = ANA_star_planner_h2.run()

    A_star_planner.timestamp.append(ANA_star_planner.timestamp[-1])
    A_star_planner_h2.timestamp.append(ANA_star_planner.timestamp[-1])
    ANA_star_planner_h2.timestamp.append(ANA_star_planner.timestamp[-1])
    ANA_star_planner_h2.total_path_cost.append(ANA_star_planner_h2.total_path_cost[-1])
    ANA_star_planner_h2.total_path_cost.append(ANA_star_planner_h2.total_path_cost[-1])
    
    ## Plot figure
    print("total runtime:", time.time() - main_start_time)
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(A_star_planner.timestamp, A_star_planner.total_path_cost, 'b-', label="A*, h: Euclidean")
    plt.plot(A_star_planner_h2.timestamp, A_star_planner_h2.total_path_cost, 'g-', label="A*, h: Custom")
    plt.plot(ANA_star_planner.timestamp[:-1], ANA_star_planner.total_path_cost, 'r-', label="ANA*, h: Euclidean")
    plt.plot(ANA_star_planner_h2.timestamp, ANA_star_planner_h2.total_path_cost, 'k-', label="ANA*, h: Custom")
    plt.title("A* and ANA* Time vs Solution Cost")
    plt.xlabel("Time (s)")
    plt.ylabel("Solution Cost")
    plt.ylim(5, 15)
    plt.legend()
    plt.show()
    print("Close the figure to continue to execute robot trajectories.")

    
    ## Visualize A* path
    if path_astar:
        print("=======================================================================")
        print("Drawing A* path with Euclidean heuristic...")
        print("=======================================================================")
        for p in path_astar:
            p = p[:2] + (0,)
            draw_sphere_marker(p, 0.05, (0, 0, 1, 1))
        execute_trajectory(robots['pr2'], base_joints, path_astar, sleep=0.2)

    if path_astar_h2:
        print("=======================================================================")
        print("Drawing A* path with custom heuristic...")
        print("=======================================================================")
        for p in path_astar_h2:
            p = p[:2] + (0,)
            draw_sphere_marker(p, 0.05, (0, 1, 0, 1))
        execute_trajectory(robots['pr2'], base_joints, path_astar_h2, sleep=0.2)

    ## Visualize ANA* path
    if path_anastar:
        print("=======================================================================")
        print("Drawing ANA* path wih Euclidean heuristic...")
        print("=======================================================================")
        for p in path_anastar:
            p = p[:2] + (0,)
            draw_sphere_marker(p, 0.05, (1, 0, 0, 1))
        execute_trajectory(robots['pr2'], base_joints, path_anastar, sleep=0.2)

    if path_anastar_h2:
        print("=======================================================================")
        print("Drawing ANA* path wih custom heuristic...")
        print("=======================================================================")
        for p in path_anastar_h2:
            p = p[:2] + (0,)
            draw_sphere_marker(p, 0.05, (0, 0, 0, 1))
        execute_trajectory(robots['pr2'], base_joints, path_anastar_h2, sleep=0.2)

    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    while True:
        main()
        print("Press enter to continue path planning for other scenarios.")
        print("Press Ctl + c to quit.")

    