#!/usr/bin/python3

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None
        self._bssf = TSPSolution(None)
        self._pruned = 0
        self._count = 0
        self._states = 0
        self._maxQ = 0

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else np.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        # Iterate through start cities
        cities = self._scenario.getCities()
        solutions = []
        for start_city in cities:
            visited_status = {city._index: False for city in cities}
            current = start_city
            path = [current]
            visited_status[current._index] = True
            next_city = min(cities, key=lambda c: current.costTo(c))
            while not all(visited_status.values()) and visited_status[next_city._index] != True:
                current = next_city
                visited_status[current._index] = True
                path.append(current)
                next_city = min(cities, key=lambda c: current.costTo(c))
            if self.is_complete_greedy(path):
                solutions.append(TSPSolution(path))
        results = {}
        # TODO other results
        results["soln"] = min(solutions, key=lambda soln: soln.cost)
        return results


    def is_complete_greedy(self, path):
        return len(path) == len(self._scenario.getCities()) \
               and len(path) == len(set(path)) \
               and path[0].costTo(path[-1]) != np.inf
    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def branchAndBound(self, time_allowance=60.0):
        results = {}
        self._pruned = 0
        self._count = 0
        self._states = 0
        self._maxQ = 1
        # Start with first problem
        # Put it on the queue
        queue = [(0, self.create_initial_problem())]

        # Initialize the best so far
        self._bssf = self.defaultRandomTour()['soln']

        start_time = time.time()
        # Repeat while queue is not empty (and we're not over time allowance)
        while len(queue) > 0 and time.time() - start_time < time_allowance:
            # Choose a problem
            chosen = self.choose(queue)

            # Expand it
            expansion = self.expand(chosen)

            for sub_prob in expansion:
                if self.is_complete_soln(sub_prob) and sub_prob.cost < self._bssf.cost:
                    self._bssf = TSPSolution(sub_prob.path)
                    self._count += 1
                elif sub_prob.lb < self._bssf.cost:
                    heapq.heappush(queue, (sub_prob.lb / len(sub_prob.path), sub_prob))
                    if len(queue) > self._maxQ:
                        self._maxQ = len(queue)
                else:
                    self._pruned += 1

        end_time = time.time()
        results['cost'] = self._bssf.cost
        results['time'] = end_time - start_time
        results['count'] = self._count
        results['soln'] = self._bssf
        results['max'] = self._maxQ
        results['total'] = self._states
        results['pruned'] = self._pruned
        return results

    def create_initial_problem(self):
        init_cost_m = np.array([[0.0] * len(self._scenario.getCities()) for _ in self._scenario.getCities()])
        start_city = None
        for city_r in self._scenario.getCities():
            i = city_r._index
            if i == 0:
                start_city = city_r
            for city_c in self._scenario.getCities():
                j = city_c._index
                init_cost_m[i][j] = city_r.costTo(city_c)
        self._states += 1
        return SubProblem(path=[start_city], parent_matrix=init_cost_m,
                          cost=0, cost_to_here=0, parent_lb=0)

    # Choose includes pruning
    def choose(self, queue):
        # sub_problem = min(queue, key=lambda p: p.lb - len(p.path))
        # queue.remove(sub_problem)
        priority, sub_problem = heapq.heappop(queue)
        while sub_problem.lb > self._bssf.cost and len(queue) > 0:
            # Pruned!
            self._pruned += 1
            # sub_problem = min(queue, key=lambda p: p.lb - len(p.path))
            # queue.remove(sub_problem)
            priority, sub_problem = heapq.heappop(queue)
        return sub_problem

    def expand(self, prob):
        cc = prob.path[-1]
        sub_probs = []

        for ind, r_cost in enumerate(prob.reduced_c_matrix[cc._index]):
            if r_cost != np.inf:
                self._states += 1
                sub_probs.append(SubProblem(prob.path + [self._scenario.getCities()[ind]],
                                            self.infinify(prob.reduced_c_matrix, cc._index, ind),
                                            prob.cost + cc.costTo(self._scenario.getCities()[ind]),
                                            r_cost,
                                            prob.lb))

        return sub_probs

    def infinify(self, matrix, row, col):
        new_m = matrix.copy()
        new_m[row] = np.inf
        new_m[:, col] = np.inf
        new_m[col][row] = np.inf
        return new_m

    def is_complete_soln(self, sub_prob):
        return len(sub_prob.path) == len(self._scenario.getCities()) and len(sub_prob.path) == len(set(sub_prob.path))



    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        pass
