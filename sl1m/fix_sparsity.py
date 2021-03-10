import numpy as np


from sl1m.constants_and_tools import *
from sl1m import planner_l1 as pl1
from sl1m import planner    as pl

from . import qp
from . import qpp

# try to import mixed integer solver
MIP_OK = False

try:
    import gurobipy
    import cvxpy as cp
    MIP_OK = True

except ImportError:
    pass


class ProblemData:
    def __init__(self, pb, success, case, res, time):
        self.pb = pb
        self.success = success
        self.case = case
        self.res = res
        self.time = time

    def __str__(self):
        return "ProblemData: \n \t problem solve success: " + str(self.success)+ "\n \t case: " + str(self.case) + "\n \t time: " + str(self.time)

    def __repr__(self):
        return self.__str__()


###########################################################
############### L1-norm minimization SOLVER ###############
###########################################################

def convertToList (C,c,A,b,E,e):
    return C.tolist(), c.tolist(), A.tolist(), b.tolist(), E.tolist(), e.tolist()

def getSurfNum (surfaces):
    numsurf = []
    for surf in surfaces:
        numsurf += [len(surf)]
    return numsurf

def getDist (allfeetpos):
    dist = 0
    from scipy.spatial import distance
    for i, pos in enumerate(allfeetpos[:-1]):
        dist += distance.euclidean(pos, allfeetpos[i+1])
    return dist


def callSolver (C,c,A,b,E,e, numsurf, goal, CPP = False, SOLVER = 0):

    if CPP and SOLVER !=2:
        C,c,A,b,E,e = convertToList(C,c,A,b,E,e)

    if SOLVER == 0: # GUROBI
        if CPP:
            res = qpp.solveLP(c,A,b,E,e,numsurf,goal)
        else:
            res = qp.solve_lp_gurobi(c,A,b,E,e)
    elif SOLVER == 1: # GLPK
        if CPP:
            res =  qpp.solveglpk(c,A,b,E,e)
        else:
            res = qp.solve_lp_glpk(c,A,b,E,e)
    elif SOLVER == 2: # GLPK
        if CPP:
            print ("GLPK is only supported in Python")
            print ("Solve Problem in Python ..")
            CPP = False
        res = qp.quadprog_solve_qp(C,c,A,b,E,e)
    return res

EPS_ = 0.01
#update cost function based on previous iteration
def reweight(x_i, c):
    return (1. / (x_i + ones(x_i.shape[0]) * EPS_)) * c
    # ~ assert norm(c2-c) > 0.1

### This solver is called when the sparsity is fixed. It assumes the first contact surface for each phase
### is the one used for contact creation.
def solve(pb, surfaces, draw_scene = None, plot = True, CPP = False, SOLVER = 0, time = 0.):
    A, b, E, e = pl.convertProblemToLp(pb)
    C = identity(A.shape[1])
    c = zeros(A.shape[1])

    # res = qpp.solveLP(c,A,b,E,e)
    # res = qp.solve_qp_gurobi(C,c,A,b,E,e)
    numsurf = []
    for phase in pb["phaseData"]:
        numsurf += [1]
        # numsurf += [len(phase["S"])]
    res = qpp.solveQP(C.tolist(),c.tolist(),A.tolist(),b.tolist(),E.tolist(),e.tolist())#, numsurf)
    # time += res.time
    print ("Time to solve QP: ", res.time)
    print ("Time to solve ALL: ", time+res.time)
    time = res.time

    if res.success:
        res = res.x
    else:
        return ProblemData(pb, False, 4, None, time)

    coms, footpos, allfeetpos = pl.retrieve_points_from_res(pb, res)

    if plot:
        ax = draw_scene(surfaces)
        pl.plotQPRes(pb, res, ax=ax)

    return ProblemData(pb, True, 0, [coms, footpos, allfeetpos], time)



def solveL1_mindist(pb, surfaces, draw_scene = None, plot = True, CPP = False, SOLVER = 0, time = 0.):
    # convert problem
    A, b, E, e = pl1.convertProblemToLp(pb, SLACK_SCALE=10.)
    c = pl1.slackSelectionMatrix(pb)
    C = identity(A.shape[1])
    C,c,A,b,E,e = convertToList(C,c,A,b,E,e)

    numsurf = []
    for phase in pb["phaseData"]:
        numsurf += [1]

    res = qpp.solveLP_mindist(c,A,b,E,e, numsurf, pb["goal"])
    print ("SUCCESS:", res.success)
    print ("MIN DIST COST:", res.cost)

    if res.success:
        res = res.x
        for i, phase in enumerate(pb["phaseData"]):
            phase["S"] = [surfaces[i][0]]

    else:
        return ProblemData(pb, False, 4, None, time)

    coms, footpos, allfeetpos = pl1.retrieve_points_from_res(pb, res)

    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)

    return solve(pb, surfaces, draw_scene, plot, CPP, SOLVER, time)
    # return ProblemData(pb, True, 0, [coms, footpos, allfeetpos], time)
    # return ProblemData(pb, True, 0, None, time)



def solveL1Reweighted(pb, surfaces, draw_scene = None, plot = True, CPP = False, SOLVER = 0, OPT = True):
    A, b, E, e = pl1.convertProblemToLp(pb, SLACK_SCALE=10.)
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)

    res = callSolver(C,c,A,b,E,e,CPP,SOLVER)
    print (res.time)
    time1 = res.time

    if res.success:
        res = res.x
    else:
        return ProblemData(pb, False, 1, None, time1)

    ok = pl1.isSparsityFixed(pb, res)
    solutionIndices = None
    solutionComb = None
    timeComb = 0.

    i = 0
    MAX_ITER = 70

    while not ok and i < MAX_ITER:
        i +=1
        c = reweight(array(res), c)
        print ("i ", i)
        res = callSolver(C,c,A,b,E,e,CPP,SOLVER)
        timeComb += res.time

        if res.success:
            res = res.x
        else:
            return ProblemData(pb, False, 10, None, time1+timeComb)
        ok = pl1.isSparsityFixed(pb, res)

    if ok:
        if plot:
            ax = draw_scene(surfaces)
            pl1.plotQPRes(pb, res, ax=ax)

        time = time1 + timeComb
        result = ProblemData(pb, True, 0, res, time)

        if True:#OPT:
            surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)
            for i, phase in enumerate(pb["phaseData"]):
                phase["S"] = [surfaces[i][indices[i]]]
            if solutionIndices is not None:
                for i, idx in enumerate(solutionIndices):
                    pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]
            # import pickle
            # with open("pb", 'wb') as f:
            #     pickle.dump(pb, f)

        # if OPT:
            # return solve(pb, surfaces, draw_scene, plot, CPP, SOLVER, time)
        # else:
            # return result
        return result

    # fail to fix sparisty
    return ProblemData(pb, False, 3, None, 0.)

### Calls the sl1m solver. Brute-forcedly tries to solve non fixed sparsity by handling the combinatorial.
### Ultimately calls solve which provides the approriate cost function
def solveL1(pb, surfaces, draw_scene = None, plot = True, CPP = False, SOLVER = 0, OPT = False):
    A, b, E, e = pl1.convertProblemToLp(pb, SLACK_SCALE=10.)
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)

    numsurf = getSurfNum(surfaces)
    res = callSolver(C,c,A,b,E,e,numsurf, pb["goal"],CPP,SOLVER)
    time1 = res.time

    if res.success:
        res = res.x
    else:
        return ProblemData(pb, False, 1, None, time1)

    ok = pl1.isSparsityFixed(pb, res)
    solutionIndices = None
    solutionComb = None
    timeComb = 0.

    if not ok:
        pbs = pl1.generateAllFixedScenariosWithFixedSparsity(pb, res)
        if pbs == None:
            return ProblemData(pb, False, 2, None, 10000.)
        for (pbComb, comb, indices) in pbs:
            A, b, E, e = pl1.convertProblemToLp(pbComb, convertSurfaces = False, SLACK_SCALE=10.)
            C = identity(A.shape[1]) * 0.00001
            c = pl1.slackSelectionMatrix(pbComb)

            numsurf = []
            for phase in pbComb["phaseData"]:
                numsurf += [len(phase["S"])]

            res = callSolver(C,c,A,b,E,e,numsurf, pb["goal"],CPP,SOLVER)
            timeComb += res.time

            if res.success:
                res = res.x
                ok = pl1.isSparsityFixed(pbComb, res)
                if ok:
                    # coms, footpos, allfeetpos = pl1.retrieve_points_from_res(pbComb, res)
                    pb = pbComb
                    solutionIndices = indices[:]
                    solutionComb = comb
                    # if plot and not OPT:
                    #     ax = draw_scene(surfaces)
                    #     pl1.plotQPRes(pb, res, ax=ax)
                    #     plot = False
                    break
            else:
                continue

    if ok:
        if plot:
            ax = draw_scene(surfaces)
            pl1.plotQPRes(pb, res, ax=ax)

        time = time1 + timeComb

        surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)
        surfaces_ = []
        for i, phase in enumerate(pb["phaseData"]):
            phase["S"] = [surfaces[i][indices[i]]]
            surfaces_ += [[surfaces[i][indices[i]]]]
            # print ([surfaces[i][indices[i]]])
        if solutionIndices is not None:
            for i, idx in enumerate(solutionIndices):
                pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]
                surfaces_[idx] = [surfaces[idx][solutionComb[i]]]

        result = ProblemData(pb, True, 0, res, time)
        if OPT:
            print ("MIN dist")
            return solveL1_mindist(pb, surfaces_, draw_scene, plot, CPP, SOLVER, time)
            # print ("OPT")
            # return solve(pb, surfaces, draw_scene, plot, CPP, SOLVER, time)
        else:
            print ("OPT x")
            return result

    # fail to fix sparisty
    return ProblemData(pb, False, 3, None, 0.)



###########################################################
################### MIXED-INTEGER SOLVER ##################
###########################################################

######################### gurobi ##########################

def solveMIP(pb, surfaces, draw_scene = None, plot = True, CPP = False, OPT = False):
    A, b, E, e = pl1.convertProblemToLp(pb, SLACK_SCALE=10.)
    c = pl1.slackSelectionMatrix(pb)
    C = identity(A.shape[1])

    if CPP:
        C,c,A,b,E,e = convertToList(C,c,A,b,E,e)
        if OPT:
            res = qpp.solveMIP_mindist(c,A,b,E,e, getSurfNum(surfaces), pb["goal"])
            print ("SOLVE MIP", res.success)
            print ("MIN DIST COST: ", res.cost)
            # res = qpp.solveMIP_QP(C,c,A,b,E,e,getSurfNum(surfaces))
            # res = qpp.solveMIP(c,A,b,E,e, getSurfNum(surfaces), pb["goal"])

            # res = qpp.solveMIP(c,A,b,E,e)
            # surfacesret, indices = pl1.bestSelectedSurfaces(pb, res.x)
            # for i, phase in enumerate(pb["phaseData"]):
            #     phase["S"] = [surfaces[i][indices[i]]]
            
            # A, b, E, e = pl.convertProblemToLp(pb)
            # C = identity(A.shape[1])
            # c = zeros(A.shape[1])
            # C,c,A,b,E,e = convertToList(C,c,A,b,E,e)
            # res = qpp.solveQP(C,c,A,b,E,e)

        else:
            res = qpp.solveMIP(c,A,b,E,e, getSurfNum(surfaces), pb["goal"])
    else:
        res = qp.solve_MIP_gurobi(c,A,b,E,e)
    time = res.time

    if res.success:
        res = res.x

        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)

        if not OPT:
            surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)
            for i, phase in enumerate(pb["phaseData"]):
                phase["S"] = [surfaces[i][indices[i]]]
            res = qpp.solveMIP_mindist(c,A,b,E,e, getSurfNum(surfaces), pb["goal"])
            print ("SOLVE MIP", res.success)
            print ("MIN DIST COST: ", res.cost)
            res = res.x
        else:
            surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)    
            for i, phase in enumerate(pb["phaseData"]):
                    phase["S"] = [surfaces[i][indices[i]]]

        # print ("OPT")
        return solve(pb, surfaces, draw_scene, plot, CPP, 0, 0)

        # # convert problem
        # A, b, E, e = pl1.convertProblemToLp(pb, SLACK_SCALE=10.)
        # c = pl1.slackSelectionMatrix(pb)
        # C = identity(A.shape[1])
        # C,c,A,b,E,e = convertToList(C,c,A,b,E,e)

        # numsurf = []
        # for phase in pb["phaseData"]:
        #     numsurf += [1]

        # res = qpp.solveMIP_mindist(c,A,b,E,e, numsurf, pb["goal"])
        # print ("SUCCESS:", res.success)
        # print ("min dist COST:", res.cost)
        # res = res.x

    else:
        # ax = draw_scene(surfaces)
        return ProblemData(pb, False, res.status, None, time)

    plot = plot and draw_scene is not None
    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)
        # pl.plotQPRes(pb, res, ax=ax)

    coms, footpos, allfeetpos = pl1.retrieve_points_from_res(pb, res)
    # coms, footpos, allfeetpos = pl.retrieve_points_from_res(pb, res)
    return ProblemData(pb, True, 0, [coms, footpos, allfeetpos], time)

def solveMIP_cost(pb, surfaces, goal,draw_scene = None, plot = True, CPP = False):
    A, b, E, e = pl1.convertProblemToLp(pb, SLACK_SCALE=10.)
    c = pl1.slackSelectionMatrix(pb)
    C = identity(A.shape[1])

    index = 0
    for i, phase in enumerate(pb["phaseData"]):
        if i == len(pb["phaseData"])-1:
            break
        index += 4
        if len(phase["S"]) != 1:
            index += 2 * len(phase["S"])

    if CPP:
        C,c,A,b,E,e = convertToList(C,c,A,b,E,e)
        res = qpp.solveMIP_cost(c,A,b,E,e,goal,index)
    else:
        res = qp.solve_MIP_gurobi(c,A,b,E,e)
    time = res.time

    if res.success:
        res = res.x
    else:
        # ax = draw_scene(surfaces)
        return ProblemData(pb, False, res.status, None, time)

    plot = plot and draw_scene is not None
    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)

    return ProblemData(pb, True, 0, res, time)

######################### cvxpy ###########################

def tovals(variables):
    return array([el.value for el in variables])

def solveMIP_cvxpy(pb, surfaces, draw_scene = None, plot = True):
    if not MIP_OK:
        print("Mixed integer formulation requires gurobi packaged in cvxpy")
        raise ImportError

    gurobipy.setParam('LogFile', '')
    gurobipy.setParam('OutputFlag', 0)

    A, b, E, e = pl1.convertProblemToLp(pb)
    slackMatrix = pl1.slackSelectionMatrix(pb)

    rdim = A.shape[1]
    varReal = cp.Variable(rdim)
    constraints = []
    constraintNormalIneq = A * varReal <= b
    constraintNormalEq   = E * varReal == e

    constraints = [constraintNormalIneq, constraintNormalEq]
    #creating boolean vars

    slackIndices = [i for i,el in enumerate (slackMatrix) if el > 0]
    numSlackVariables = len([el for el in slackMatrix if el > 0])
    boolvars = cp.Variable(numSlackVariables, boolean=True)
    obj = cp.Minimize(slackMatrix * varReal)

    if MIP:
        constraints = constraints + [varReal[el] <= 100. * boolvars[i] for i, el in enumerate(slackIndices)]

        currentSum = []
        previousL = 0
        for i, el in enumerate(slackIndices):
            if i!= 0 and el - previousL > 2.:
                assert len(currentSum) > 0
                constraints = constraints + [sum(currentSum) == len(currentSum) -1 ]
                currentSum = [boolvars[i]]
            elif el !=0:
                currentSum = currentSum + [boolvars[i]]
            previousL  = el
        if len(currentSum) > 1:
            constraints = constraints + [sum(currentSum) == len(currentSum) -1 ]
    obj = cp.Minimize(ones(numSlackVariables) * boolvars)
    prob = cp.Problem(obj, constraints)
    t1 = clock()
    res = prob.solve(solver=cp.GUROBI, verbose=False )
    t2 = clock()
    res = tovals(varReal)
    print("time to solve MIP ", timMs(t1,t2))


    plot = plot and draw_scene is not None
    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)

    return ProblemData(pb, True, 0, res, timMs(t1,t2))
