import numpy as np

from sl1m.constants_and_tools import *
from sl1m import planner_l1 as pl1
from sl1m import planner    as pl

from sl1m import qp
from sl1m import qpg
import boost

from time import clock

# try to import mixed integer solver
MIP_OK = False  
try:
    import gurobipy
    import cvxpy as cp
    MIP_OK = True

except ImportError:
    pass


np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
DEFAULT_NUM_VARS = 4

EPS = 0.001
def reweight (x, c):
    return (1. / (x + ones(x.shape[0]) * EPS)) * c


### SOLVE FUNCTIONS
### This solver is called when the sparsity is fixed. It assumes the first contact surface for each phase
### is the one used for contact creation.

def solve_glpk(pb, surfaces, draw_scene = None, plot = True, time = 0., cpp = False):  
        
    A, b, E, e = pl.convertProblemToLp(pb)    
    C = identity(A.shape[1])
    c = zeros(A.shape[1])
    time1 = 0.

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()
        res = qpg.solveglpk(c,A,b,E,e)
        time1 = res.time

    else:
        t1 = clock()
        res = qp.solve_lp_glpk(c,A,b,E,e)
        t2 = clock()
        time1 = timMs(t1, t2)
    
    if res.success:
        res = res.x
    else:
        print ("CASE3: turned out to be infeasible")
        return 3,3,3
    
    print "TOTAL solve time"                 , time1+time
    
    coms, footpos, allfeetpos = pl.retrieve_points_from_res(pb, res)
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl.plotQPRes(pb, res, ax=ax)
    
    return pb, res, time1+time


def solve(pb,surfaces, draw_scene = None, plot = True, time = 0., cpp = False):  
        
    A, b, E, e = pl.convertProblemToLp(pb)    
    C = identity(A.shape[1])
    c = zeros(A.shape[1])
    time1 = 0.

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()    
        res = qpg.solveLP(c,A,b,E,e)
        time1 = res.time
    else:
        t1 = clock()
        res = qp.solve_lp_gurobi(c,A,b,E,e)
        t2 = clock()
        time1 = timMs(t1, t2)
    
    if res.success:
        res = res.x
    else:
        print ("CASE3: turned out to be infeasible")
        return 3, 3, 3
    
    print "TOTAL solve time"                 , time1+time
    
    coms, footpos, allfeetpos = pl.retrieve_points_from_res(pb, res)
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl.plotQPRes(pb, res, ax=ax)
    
    return pb, res, time+time1


def solve_cost(pb,surfaces, draw_scene = None, plot = True, time = 0., cpp = False):  
        
    A, b, E, e = pl.convertProblemToLp(pb)    
    C = identity(A.shape[1])
    c = zeros(A.shape[1])
    nVarEnd = pl.numVariablesForPhase(pb["phaseData"][-1])
    time1 = 0.

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()    
        res = qpg.solveLP_cost(c,A,b,E,e,nVarEnd,pb["goal"][1],1.0) #weight 1.0
        time1 = res.time
    else:
        t1 = clock()
        res = qp.solve_lp_gurobi_cost(c,A,b,E,e,nVarEnd,pb["goal"][1],1.0) #weight 1.0
        t2 = clock()
        time1 = timMs(t1, t2)
    
    if res.success:
        res = res.x
    else:
        print ("CASE3: turned out to be infeasible")
        return 3, 3, 3
    
    coms, footpos, allfeetpos = pl.retrieve_points_from_res(pb, res)
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl.plotQPRes(pb, res, ax=ax)
    
    return pb, res, time + time1
    


### SOLVE LP FUNCTIONS
### Calls the sl1m solver. Brute-forcedly tries to solve non fixed sparsity by handling the combinatorial.
### Ultimately calls solve which provides the approriate cost function

def solveL1_glpk(pb, surfaces, draw_scene = None, plot = True, cpp = False):     
    A, b, E, e = pl1.convertProblemToLp(pb)    
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)
    time1 = 0.

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()  
        res = qpg.solveglpk(c,A,b,E,e)
        time1 = res.time
    else:
        t1 = clock()
        res = qp.solve_lp_glpk(c,A,b,E,e)
        t2 = clock()
        time1 = timMs(t1, t2)

    if res.success:
        res = res.x
    else:
        print ("CASE4: fail to make the first guess")
        return 4,4,4
    
    #print "time to solve lp ", time1
        
    ok = pl1.isSparsityFixed(pb, res)
    solutionIndices = None
    solutionComb = None
    pbs = None
    timeComb = 0.
    
    if not ok:
        # print "SOLVE COMB"
        pbs = pl1.generateAllFixedScenariosWithFixedSparsity(pb, res)

        if pbs == 1:
            print "CASE1: too big combinatorial"
            return 1, 1, 1
        
        for (pbComb, comb, indices) in pbs:
            A, b, E, e = pl1.convertProblemToLp(pbComb, convertSurfaces = False)
            C = identity(A.shape[1]) * 0.00001
            c = pl1.slackSelectionMatrix(pbComb)

            if cpp:
                A = A.tolist()
                b = b.tolist()
                E = E.tolist()
                e = e.tolist()
                c = c.tolist()
                res = qpg.solveglpk(c,A,b,E,e)
                timeComb += res.time
            else:
                t3 = clock()
                res = qp.solve_lp_glpk(c,A,b,E,e)
                t4 = clock()
                timeComb += timMs(t3, t4)

            if res.success:
                res = res.x
                if pl1.isSparsityFixed(pbComb, res):       
                    coms, footpos, allfeetpos = pl1.retrieve_points_from_res(pbComb, res)
                    pb = pbComb
                    ok = True
                    solutionIndices = indices[:]
                    solutionComb = comb
                    break
            else:
                continue
             
    time = time1+timeComb
    
    if ok:
        if plot:
            ax = draw_scene(surfaces)
            pl1.plotQPRes(pb, res, ax=ax)

        surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)        
        for i, phase in enumerate(pb["phaseData"]): 
            phase["S"] = [surfaces[i][indices[i]]]
        if solutionIndices is not None:
            for i, idx in enumerate(solutionIndices):
                pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]

        return solve_glpk (pb, surfaces, draw_scene, plot, time, cpp)  
    
    print "CASE2: combinatorials all sparsity not fixed"
    return 2, 2, 2   

def solveL1(pb, surfaces, draw_scene = None, plot = True, cpp = False):     
    A, b, E, e = pl1.convertProblemToLp(pb)    
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)
    time1 = 0.

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()    
        res = qpg.solveLP(c,A,b,E,e)
        time1 = res.time
    else:
        t1 = clock()
        res = qp.solve_lp_gurobi(c,A,b,E,e)
        t2 = clock()
        time1 = timMs(t1, t2)

    if res.success:
        res = res.x
    else:
        print ("CASE4: fail to make the first guess")
        return 4,4,4
        
    ok = pl1.isSparsityFixed(pb, res)
    solutionIndices = None
    solutionComb = None
    pbs = None
    timeComb = 0.
    
    if not ok:
        pbs = pl1.generateAllFixedScenariosWithFixedSparsity(pb, res)

        if pbs == 1:
            print "CASE1: too big combinatorial"
            return 1, 1, 1
        
        for (pbComb, comb, indices) in pbs:
            A, b, E, e = pl1.convertProblemToLp(pbComb, convertSurfaces = False)
            C = identity(A.shape[1]) * 0.00001
            c = pl1.slackSelectionMatrix(pbComb)

            if cpp:
                A = A.tolist()
                b = b.tolist()
                E = E.tolist()
                e = e.tolist()
                c = c.tolist()
                res = qpg.solveLP(c,A,b,E,e)
                timeComb += res.time
            else:
                t3 = clock()
                res = qp.solve_lp_gurobi(c,A,b,E,e)
                t4 = clock()
                timeComb += timMs(t3, t4)

            if res.success:
                res = res.x
                if pl1.isSparsityFixed(pbComb, res):       
                    coms, footpos, allfeetpos = pl1.retrieve_points_from_res(pbComb, res)
                    pb = pbComb
                    ok = True
                    solutionIndices = indices[:]
                    solutionComb = comb
                    break
            else:
                continue
            
    time = time1+timeComb
    
    if ok:
        # surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)        
        # for i, phase in enumerate(pb["phaseData"]): 
        #     phase["S"] = [surfaces[i][indices[i]]]
        # if solutionIndices is not None:
        #     for i, idx in enumerate(solutionIndices):
        #         pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]
        
        if plot:
            ax = draw_scene(surfaces)
            pl1.plotQPRes(pb, res, ax=ax)

        # return solve (pb, surfaces, draw_scene, plot, time, cpp)  
        return pb, res, time
    
    print "CASE2: combinatorials all sparsity not fixed"
    return 2, 2, 2   

def solveL1_re(pb, surfaces, draw_scene = None, plot = True, cpp = False):     
    A, b, E, e = pl1.convertProblemToLp(pb)    
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)
    time1 = 0.; model = None

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()    
        res = qpg.solveLP_init(c,A,b,E,e)
        time1 = res.time
    else:
        t1 = clock()
        res = qp.solve_lp_gurobi(c,A,b,E,e)
        t2 = clock()
        time1 = timMs(t1, t2)
        
        if res.success:
            model = res.model

    if res.success:
        res = res.x
    else:
        print ("CASE4: fail to make the first guess")
        return 4,4,4,4
        
    ok = pl1.isSparsityFixed(pb, res)
    timeComb = 0.

    prev_res = res
    i = 0
    MAX_ITER = 15
    
    while not ok and i < MAX_ITER:
        # print i, "th ITER"
        i +=1        
        c = reweight(array(prev_res), c)
        if cpp:
            c = c.tolist()    
            res = qpg.solveLP_iter(c,A,b,E,e)
            timeComb += res.time

        else:   
            t3 = clock()
            res = qp.solve_lp_gurobi_iter(model,c,A,b,E,e)
            t4 = clock()
            timeComb += timMs(t3, t4)

            if res.success: 
                model = res.model

        if res.success:
            print "CONVERGED"
            res = res.x
            prev_res = res
        else: 
            print "DID NOT CONVERGED"
            continue      

        ok = pl1.isSparsityFixed(pb,res)
        if not ok:
            print "CONVERGED BUT SPARSITY NOT FIXED"

    time = time1+timeComb
    iteration = i
    
    if ok:
        # surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)        
        # for i, phase in enumerate(pb["phaseData"]): 
        #     phase["S"] = [surfaces[i][indices[i]]]
        # if solutionIndices is not None:
        #     for i, idx in enumerate(solutionIndices):
        #         pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]
        
        if plot:
            ax = draw_scene(surfaces)
            pl1.plotQPRes(pb, res, ax=ax)

        # return solve (pb, surfaces, draw_scene, plot, time, cpp)  
        return pb, res, time, iteration
    
    print "CASE2: combinatorials all sparsity not fixed"
    return 2, 2, 2 ,2

def solveL1_cost(pb, surfaces, draw_scene = None, plot = True, cpp = False, weight = 0., linear = False):     
    A, b, E, e = pl1.convertProblemToLp(pb)    
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)
    nVarEnd = pl1.numVariablesForPhase(pb["phaseData"][-1])
    time1 = 0.

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()    
        res = qpg.solveLP_cost(c,A,b,E,e,nVarEnd,pb["goal"][1].tolist(),weight)
        time1 = res.time
    else:
        t1 = clock()
        res = qp.solve_lp_gurobi_cost(c,A,b,E,e,nVarEnd,pb["goal"][1],weight,linear)
        t2 = clock()
        time1 = timMs(t1, t2)

    if res.success:
        res = res.x
    else:
        print ("CASE4: fail to make the first guess")
        return 4,4,4
            
    ok = pl1.isSparsityFixed(pb, res)
    solutionIndices = None
    solutionComb = None
    pbs = None
    timeComb = 0.
    
    if not ok:
        pbs = pl1.generateAllFixedScenariosWithFixedSparsity(pb, res)

        if pbs == 1:
            print "CASE1: too big combinatorial"
            return 1, 1, 1
        
        for (pbComb, comb, indices) in pbs:
            A, b, E, e = pl1.convertProblemToLp(pbComb, convertSurfaces = False)
            C = identity(A.shape[1]) * 0.00001
            c = pl1.slackSelectionMatrix(pbComb)

            if cpp:
                A = A.tolist()
                b = b.tolist()
                E = E.tolist()
                e = e.tolist()
                c = c.tolist()
                res = qpg.solveLP_cost(c,A,b,E,e,nVarEnd,pb["goal"][1].tolist(),weight)
                timeComb += res.time
            else:
                t3 = clock()
                res = qp.solve_lp_gurobi_cost(c,A,b,E,e,nVarEnd,pb["goal"][1].tolist(),weight)
                t4 = clock()
                timeComb += timMs(t3, t4)

            if res.success:
                res = res.x
                if pl1.isSparsityFixed(pbComb, res):       
                    coms, footpos, allfeetpos = pl1.retrieve_points_from_res(pbComb, res)
                    pb = pbComb
                    ok = True
                    solutionIndices = indices[:]
                    solutionComb = comb
                    break
            else:
                continue
            
    time = time1+timeComb
    
    if ok:
        # surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)        
        # for i, phase in enumerate(pb["phaseData"]): 
        #     phase["S"] = [surfaces[i][indices[i]]]
        # if solutionIndices is not None:
        #     for i, idx in enumerate(solutionIndices):
        #         pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]
        
        if plot:
            ax = draw_scene(surfaces)
            pl1.plotQPRes(pb, res, ax=ax)

        # return solve_cost (pb, surfaces, draw_scene, plot, time, cpp)  
        return pb, res, time
    
    print "CASE2: combinatorials all sparsity not fixed"
    return 2, 2, 2   

def solveL1_cost_re(pb, surfaces, draw_scene = None, plot = True, cpp = False, weight = 0., linear = False):     
    A, b, E, e = pl1.convertProblemToLp(pb)    
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)
    nVarEnd = pl1.numVariablesForPhase(pb["phaseData"][-1])
    time1 = 0.; model = None

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()    
        res = qpg.solveLP_cost_init(c,A,b,E,e,nVarEnd,pb["goal"][1].tolist(),weight)
        time1 = res.time
    else:
        t1 = clock()
        res = qp.solve_lp_gurobi_cost(c,A,b,E,e,nVarEnd,pb["goal"][1],weight,linear)
        t2 = clock()
        time1 = timMs(t1, t2)

        if res.success:
            model = res.model

    if res.success:
        res = res.x
    else:
        print ("CASE4: fail to make the first guess")
        return 4,4,4,4
            
    ok = pl1.isSparsityFixed(pb, res)
    timeComb = 0.
    
    prev_res = res
    i = 0
    MAX_ITER = 15
    
    while not ok and i < MAX_ITER:
        # print i, "th ITER"
        i +=1        
        c = reweight(array(prev_res), c)
        if cpp:
            c = c.tolist()    
            res = qpg.solveLP_cost_iter(c,A,b,E,e,nVarEnd,pb["goal"][1].tolist(),weight)
            timeComb += res.time

        else:   
            t3 = clock()
            res = qp.solve_lp_gurobi_cost_iter(model,c,A,b,E,e,nVarEnd,pb["goal"][1].tolist(),weight,linear)
            t4 = clock()
            timeComb += timMs(t3, t4)

            if res.success: 
                model = res.model

        if res.success:
            res = res.x
            prev_res = res
        else: 
            continue      

        ok = pl1.isSparsityFixed(pb,res)

    time = time1+timeComb
    iteration = i
    
    if ok:
        # surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)        
        # for i, phase in enumerate(pb["phaseData"]): 
        #     phase["S"] = [surfaces[i][indices[i]]]
        # if solutionIndices is not None:
        #     for i, idx in enumerate(solutionIndices):
        #         pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]
        
        if plot:
            ax = draw_scene(surfaces)
            pl1.plotQPRes(pb, res, ax=ax)

        # return solve_cost (pb, surfaces, draw_scene, plot, time, cpp)  
        return pb, res, time, iteration
    
    print "CASE2: combinatorials all sparsity not fixed"
    return 2, 2, 2, 2
        
### MIXED-INTEGER SOLVERS

def tovals(variables):
    return array([el.value for el in variables])

def solveMIP_cp(pb, surfaces, MIP = True, draw_scene = None, plot = True):  
    if not MIP_OK:
        print "Mixed integer formulation requires gurobi packaged in cvxpy"
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
    # print "time to solve MIP ", timMs(t1,t2)
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)
    
    #return timMs(t1,t2)
    return pb, res, timMs(t1, t2)

def solveMIP(pb, surfaces, MIP = True, draw_scene = None, plot = True, cpp=False, wslack=False):  
       
    A, b, E, e = pl1.convertProblemToLp(pb)   
    c = pl1.slackSelectionMatrix(pb)  

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()
        res = qpg.solveMIP(c,A,b,E,e,wslack)
        time = res.time
    else:
        t1 = clock()
        res = qp.solve_MIP_gurobi(c,A,b,E,e,wslack)
        t2 = clock()
        time = timMs(t1, t2)

    if res.success:
        res = res.x
    else:
        print ("GUROBI FAIL")
        return 0, 0, 0
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)
    
    return pb, res, time

def solveMIP_cost(pb, surfaces, MIP = True, draw_scene = None, plot = True, cpp=False, wslack=False, linear=False):  
       
    A, b, E, e = pl1.convertProblemToLp(pb)   
    c = pl1.slackSelectionMatrix(pb)  
    nVarEnd = pl1.numVariablesForPhase(pb["phaseData"][-1])

    if cpp:
        A = A.tolist()
        b = b.tolist()
        E = E.tolist()
        e = e.tolist()
        c = c.tolist()
        res = qpg.solveMIP_cost(c,A,b,E,e, nVarEnd, pb["goal"][1].tolist(),wslack)
        time = res.time
    else:
        t1 = clock()
        res = qp.solve_MIP_gurobi_cost(c,A,b,E,e,nVarEnd, pb["goal"][1],wslack,linear)
        t2 = clock()
        time = timMs(t1, t2)

    if res.success:
        res = res.x
    else:
        print ("GUROBI FAIL")
        return 0, 0, 0
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)
    
    return pb, res, time
   