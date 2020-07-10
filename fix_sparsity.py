import numpy as np


from sl1m.constants_and_tools import *
from sl1m import planner_l1 as pl1
from sl1m import planner    as pl

from sl1m import qp
from sl1m import qpg
import boost

# try to import mixed integer solver
MIP_OK = False  
try:
    import gurobipy
    import cvxpy as cp
    MIP_OK = True

except ImportError:
    pass


from time import clock

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
DEFAULT_NUM_VARS = 4

##### pareto
    
import pickle

def readFromFile (fileName):
  data = []
  try:
      with open(fileName,'rb') as f:
        while True:
          try:
            line = pickle.load(f)
          except EOFError:
            break
          data.append(line)  
  except:
      return None
  return data[0]


### This solver is called when the sparsity is fixed. It assumes the first contact surface for each phase
### is the one used for contact creation.
#def solve(pb,surfaces, draw_scene = None, plot = True):  
def solve_glpk(pb, surfaces, draw_scene = None, plot = True, time = 0.):  
        
    t1 = clock()
    A, b, E, e = pl.convertProblemToLp(pb)    
    C = identity(A.shape[1])
    c = zeros(A.shape[1])
    
    
    A = A.tolist()
    b = b.tolist()
    E = E.tolist()
    e = e.tolist()
    c = c.tolist()
    
    #res = qp.quadprog_solve_qp(C, c,A,b,E,e)    ######
    #res = qp.solve_lp_gurobi(c,A,b,E,e)
    
    res = qpg.solveglpk(c,A,b,E,e)
    #res = qpg.solveglpk(c,A,b,E,e)
    #if res.success:
        #time1 = res.time
        #res = res.x
    #else:
        #print ("CASE3: turned out to be infeasible")
        #return 3, 3, 3
    time1 = res[0]
    del res[0]
    res = array(res)
    
    #print "time to set up problem" , timMs(t1,t2)
    #print "time to solve problem"  , timMs(t2,t3)
    #print "total time"             , timMs(t1,t3)
    print "solve time"                 , time1
    print "TOTAL solve time"                 , time1+time
    
    coms, footpos, allfeetpos = pl.retrieve_points_from_res(pb, res)
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl.plotQPRes(pb, res, ax=ax)
    
    return pb, res, time1+time
    #return timMs(t1,t3)+time
    # return pb, coms, footpos, allfeetpos, res
    
def solve(pb,surfaces, draw_scene = None, plot = True, time = 0.):  
        
    t1 = clock()
    A, b, E, e = pl.convertProblemToLp(pb)    
    C = identity(A.shape[1])
    c = zeros(A.shape[1])
    t2 = clock()
    #res = qp.quadprog_solve_qp(C, c,A,b,E,e)    ######
    res = qp.solve_lp_gurobi(c,A,b,E,e)
    #res = array(qpg.solveLP(c,A,b,E,e))
    
    if res.success:
        #time1 = res.time
        res = res.x
    else:
        print ("CASE3: turned out to be infeasible")
        return 3, 3, 3
    t3 = clock()
    
    #print "time to set up problem" , timMs(t1,t2)
    #print "time to solve problem"  , timMs(t2,t3)
    #print "total time"             , timMs(t1,t3)
    #print "lp time"                 , time
    #print "solve time"             , time1
    #print "total time"             , time+time1
    
    coms, footpos, allfeetpos = pl.retrieve_points_from_res(pb, res)
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl.plotQPRes(pb, res, ax=ax)
    
    return pb, res, 0#time1+time
    #return timMs(t1,t3)+time
    # return pb, coms, footpos, allfeetpos, res

def solve_gr(pb,surfaces, draw_scene = None, plot = True, time = 0.):  
        
    t1 = clock()
    A, b, E, e = pl.convertProblemToLp(pb)    
    C = identity(A.shape[1])
    c = zeros(A.shape[1])
    
    
    A = A.tolist()
    b = b.tolist()
    E = E.tolist()
    e = e.tolist()
    c = c.tolist()
    
    #res = qp.quadprog_solve_qp(C, c,A,b,E,e)    ######
    #res = qp.solve_lp_gurobi(c,A,b,E,e)
    
    res = qpg.solveLP(c,A,b,E,e)
    #res = qpg.solveglpk(c,A,b,E,e)
    #if res.success:
        #time1 = res.time
        #res = res.x
    #else:
        #print ("CASE3: turned out to be infeasible")
        #return 3, 3, 3
    time1 = res[0]
    del res[0]
    res = array(res)
    
    #print "time to set up problem" , timMs(t1,t2)
    #print "time to solve problem"  , timMs(t2,t3)
    #print "total time"             , timMs(t1,t3)
    print "solve time"                 , time1
    print "TOTAL solve time"                 , time1+time
    
    coms, footpos, allfeetpos = pl.retrieve_points_from_res(pb, res)
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl.plotQPRes(pb, res, ax=ax)
    
    return pb, res, time1+time
    #return timMs(t1,t3)+time
    # return pb, coms, footpos, allfeetpos, res

def solve_gr_cost(pb,surfaces, draw_scene = None, plot = True, time = 0., weight = 0., linear = False):  
        
    t1 = clock()
    A, b, E, e = pl.convertProblemToLp(pb)    
    C = identity(A.shape[1])
    c = zeros(A.shape[1])
    
    t2 = clock()
    nVarEnd = pl.numVariablesForPhase(pb["phaseData"][-1])
    res = qp.qp.solve_lp_gurobi_cost(c,nVarEnd,pb["goal"][1],A,b,E,e,weight,linear)
    t3 = clock()

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
    
    return pb, res, time + t3-t1
    

def solveL1(pb, surfaces, draw_scene = None, plot = True):     
    A, b, E, e = pl1.convertProblemToLp(pb)    
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)
        
    t1 = clock()
    #res = qp.quadprog_solve_qp(C, c,A,b,E,e)
    res = qp.solve_lp_gurobi(c,A,b,E,e)
    # ~ res = qp.solve_lp_gurobi(c,A,b,E,e).x
    # ~ res = qp.solve_lp_glpk(c,A,b,E,e).x
    t2 = clock()
    time1 = t2-t1
    
    if res.success:
        #time1 = res.time
        res = res.x
    else:
        print ("CASE4: fail to make the first guess")
        return 4,4,4
    
    #print "time to solve lp ", timMs(t1,t2)
        
    ok = pl1.isSparsityFixed(pb, res)
    solutionIndices = None
    solutionComb = None
    pbs = None
    timeComb = 0.
    
    if not ok:
        print "SOLVE COMB"
        pbs = pl1.generateAllFixedScenariosWithFixedSparsity(pb, res)
        
        t3 = clock()
        
        if pbs == 1:
            print "CASE1: too big combinatorial"
            return 1, 1, 1
        
        for (pbComb, comb, indices) in pbs:
            A, b, E, e = pl1.convertProblemToLp(pbComb, convertSurfaces = False)
            C = identity(A.shape[1]) * 0.00001
            c = pl1.slackSelectionMatrix(pbComb)
            res = qp.quadprog_solve_qp(C, c,A,b,E,e)
            #res = qp.solve_lp_gurobi(c,A,b,E,e)
            #timeComb += res.time
            if res.success:
                res = res.x
                if pl1.isSparsityFixed(pbComb, res):       
                    coms, footpos, allfeetpos = pl1.retrieve_points_from_res(pbComb, res)
                    pb = pbComb
                    ok = True
                    solutionIndices = indices[:]
                    solutionComb = comb
                    if plot:
                        ax = draw_scene(surfaces)
                        pl1.plotQPRes(pb, res, ax=ax)
                    break
            else:
                print "unfeasible problem"
                pass
            
        t4 = clock()      
        #print "time to solve combinatorial ", timMs(t3,t4)
    time = time1+timeComb
    
    if ok:
        surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)        
        for i, phase in enumerate(pb["phaseData"]): 
            phase["S"] = [surfaces[i][indices[i]]]
        if solutionIndices is not None:
            for i, idx in enumerate(solutionIndices):
                pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]
        
        #return solve (pb, surfaces, draw_scene, plot)  
        return solve (pb, surfaces, draw_scene, plot, time)  
    
    print "CASE2: combinatorials all sparsity not fixed"
    return 2, 2, 2   
        
        # return solve(pb,surfaces, draw_scene = draw_scene, plot = True )  
       

### Calls the sl1m solver. Brute-forcedly tries to solve non fixed sparsity by handling the combinatorial.
### Ultimately calls solve which provides the approriate cost function
def solveL1_glpk(pb, surfaces, draw_scene = None, plot = True):     
    A, b, E, e = pl1.convertProblemToLp(pb)    
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)
    
    A = A.tolist()
    b = b.tolist()
    E = E.tolist()
    e = e.tolist()
    c = c.tolist()    
    #t1 = clock()
    res = qpg.solveglpk(c,A,b,E,e)
    #res = qp.quadprog_solve_qp(C, c,A,b,E,e)
    #res = qp.solve_lp_gurobi(c,A,b,E,e)
    # ~ res = qp.solve_lp_gurobi(c,A,b,E,e).x
    # ~ res = qp.solve_lp_glpk(c,A,b,E,e).x
    #t2 = clock()
    time1 = res[0]
    del res[0]
    res = array(res)
    
    #if res.success:
        #time1 = res.time
        #res = res.x
    #else:
        #print ("CASE4: fail to make the first guess")
        #return 4,4,4
    
    #print "time to solve lp ", time1
    #print "time to solve lp ", timMs(t1,t2)
        
    ok = pl1.isSparsityFixed(pb, res)
    solutionIndices = None
    solutionComb = None
    pbs = None
    timeComb = 0.
    
    if not ok:
        print "SOLVE COMB"
        pbs = pl1.generateAllFixedScenariosWithFixedSparsity(pb, res)

        if pbs == 1:
            print "CASE1: too big combinatorial"
            return 1, 1, 1
        
        for (pbComb, comb, indices) in pbs:
            A, b, E, e = pl1.convertProblemToLp(pbComb, convertSurfaces = False)
            C = identity(A.shape[1]) * 0.00001
            c = pl1.slackSelectionMatrix(pbComb)
            #res = qp.quadprog_solve_qp(C, c,A,b,E,e)
            #res = qp.solve_lp_gurobi(c,A,b,E,e)
            A = A.tolist()
            b = b.tolist()
            E = E.tolist()
            e = e.tolist()
            c = c.tolist()
            t3 = clock()
            res = qpg.solveglpk(c,A,b,E,e)
            t4 = clock()
            timeComb += res[0]
            del res[0]
            res = array(res)
            #if res.success:
                #res = res.x
            if pl1.isSparsityFixed(pbComb, res):       
                coms, footpos, allfeetpos = pl1.retrieve_points_from_res(pbComb, res)
                pb = pbComb
                ok = True
                solutionIndices = indices[:]
                solutionComb = comb
                if plot:
                    ax = draw_scene(surfaces)
                    pl1.plotQPRes(pb, res, ax=ax)
                break
            #else:
                #print "unfeasible problem"
                #pass
             
        #timeComb = timMs(t3,t4)
        #print "time to solve combinatorial ", timeComb
        #print "time to solve combinatorial ", timMs(t3,t4)
    time = time1+timeComb
    
    if ok:
        surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)        
        for i, phase in enumerate(pb["phaseData"]): 
            phase["S"] = [surfaces[i][indices[i]]]
        if solutionIndices is not None:
            for i, idx in enumerate(solutionIndices):
                pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]
        
        #return solve (pb, surfaces, draw_scene, plot)  
        return solve_glpk (pb, surfaces, draw_scene, plot, time)  
    
    print "CASE2: combinatorials all sparsity not fixed"
    return 2, 2, 2   
        
        # return solve(pb,surfaces, draw_scene = draw_scene, plot = True ) 

def solveL1_gr(pb, surfaces, draw_scene = None, plot = True):     
    A, b, E, e = pl1.convertProblemToLp(pb)    
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)
    
    A = A.tolist()
    b = b.tolist()
    E = E.tolist()
    e = e.tolist()
    c = c.tolist()    
    #t1 = clock()
    res = qpg.solveLP(c,A,b,E,e)
    #res = qpg.solveglpk(c,A,b,E,e)
    #res = qp.quadprog_solve_qp(C, c,A,b,E,e)
    #res = qp.solve_lp_gurobi(c,A,b,E,e)
    # ~ res = qp.solve_lp_gurobi(c,A,b,E,e).x
    # ~ res = qp.solve_lp_glpk(c,A,b,E,e).x
    #t2 = clock()
    time1 = res[0]
    del res[0]
    res = array(res)
    
    #if res.success:
        #time1 = res.time
        #res = res.x
    #else:
        #print ("CASE4: fail to make the first guess")
        #return 4,4,4
    
    #print "time to solve lp ", time1
    #print "time to solve lp ", timMs(t1,t2)
        
    ok = pl1.isSparsityFixed(pb, res)
    solutionIndices = None
    solutionComb = None
    pbs = None
    timeComb = 0.
    
    if not ok:
        print "SOLVE COMB"
        pbs = pl1.generateAllFixedScenariosWithFixedSparsity(pb, res)

        if pbs == 1:
            print "CASE1: too big combinatorial"
            return 1, 1, 1
        
        for (pbComb, comb, indices) in pbs:
            A, b, E, e = pl1.convertProblemToLp(pbComb, convertSurfaces = False)
            C = identity(A.shape[1]) * 0.00001
            c = pl1.slackSelectionMatrix(pbComb)
            #res = qp.quadprog_solve_qp(C, c,A,b,E,e)
            #res = qp.solve_lp_gurobi(c,A,b,E,e)
            A = A.tolist()
            b = b.tolist()
            E = E.tolist()
            e = e.tolist()
            c = c.tolist()
            t3 = clock()
            res = qpg.solveLP(c,A,b,E,e)
            #res = qpg.solveglpk(c,A,b,E,e)
            t4 = clock()
            timeComb += res[0]
            del res[0]
            res = array(res)
            #if res.success:
                #res = res.x
            if pl1.isSparsityFixed(pbComb, res):       
                coms, footpos, allfeetpos = pl1.retrieve_points_from_res(pbComb, res)
                pb = pbComb
                ok = True
                solutionIndices = indices[:]
                solutionComb = comb
                if plot:
                    ax = draw_scene(surfaces)
                    pl1.plotQPRes(pb, res, ax=ax)
                break
            #else:
                #print "unfeasible problem"
                #pass
             
        #timeComb = timMs(t3,t4)
        #print "time to solve combinatorial ", timeComb
        #print "time to solve combinatorial ", timMs(t3,t4)
    time = time1+timeComb
    
    if ok:
        surfacesret, indices = pl1.bestSelectedSurfaces(pb, res)        
        for i, phase in enumerate(pb["phaseData"]): 
            phase["S"] = [surfaces[i][indices[i]]]
        if solutionIndices is not None:
            for i, idx in enumerate(solutionIndices):
                pb["phaseData"][idx]["S"] = [surfaces[idx][solutionComb[i]]]
        
        #return solve (pb, surfaces, draw_scene, plot)  
        return solve_gr (pb, surfaces, draw_scene, plot, time)  
        #return pb, res, time
    
    print "CASE2: combinatorials all sparsity not fixed"
    return 2, 2, 2   
        
        # return solve(pb,surfaces, draw_scene = draw_scene, plot = True )  

EPS = 0.001

def reweight (x, c):
    return (1. / (x + ones(x.shape[0]) * EPS)) * c

def solveL1_gr_cost(pb, surfaces, draw_scene = None, plot = True, weight=0., linear=False):     
    A, b, E, e = pl1.convertProblemToLp(pb)    
    C = identity(A.shape[1]) * 0.00001
    c = pl1.slackSelectionMatrix(pb)
    #c = pl1.slackSelectionMatrix_cost(pb, neglected_surfs)
    nVarEnd = pl1.numVariablesForPhase(pb["phaseData"][-1])
        
    t1 = clock()
    res = qp.solve_lp_gurobi_cost_pre(c,A,b,E,e,nVarEnd,pb["goal"][1],weight,linear)
    t2 = clock()

    time1 = t2-t1 # time to solve first guess
    
    if res.success:
        #time1 = res.time
        tmp = res
        model = res.model
        res = res.x
    else:
        print ("CASE4: fail to make the first guess")
        return 4,4,4
        
        
    ok = pl1.isSparsityFixed(pb, res)
    solutionIndices = None
    solutionComb = None
    pbs = None
    timeComb = 0.
    SOLVECOMB = False
    
    
    ## iterative reweighting
    i = 0
    MAX_ITER = 20
    t3 = clock()
    while not ok and i <MAX_ITER:
        print i, "th ITER"
        SOLVECOMB = True
        i +=1        
        c = reweight(array(res), c)
        res = qp.solve_lp_gurobi_cost(model,c,A,b,E,e,nVarEnd,pb["goal"][1],weight,linear).x
        ok = pl1.isSparsityFixed(pb,res)

    t4 = clock()
    time2 = t4-t3 # time to solve combinatorial
    
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
        
        # return solve(pb, surfaces, draw_scene, plot, time1+time2, weight, linear)
        return pb, res, time1+time2
    
    print "CASE2: combinatorials all sparsity not fixed"
    return 2, 2, 2   
        
############### MIXED-INTEGER SOLVER ###############

def tovals(variables):
    return array([el.value for el in variables])

def solveMIP_gr_cost(pb, surfaces, MIP = True, draw_scene = None, plot = True,linear=False):  
    if not MIP_OK:
        print "Mixed integer formulation requires gurobi packaged in cvxpy"
        raise ImportError
        
    gurobipy.setParam('LogFile', '')
    gurobipy.setParam('OutputFlag', 0)
       
    A, b, E, e = pl1.convertProblemToLp(pb)   
    #slackMatrix = pl1.slackSelectionMatrix(pb)
    
    ###
    c = pl1.slackSelectionMatrix(pb)
    #c = pl1.slackSelectionMatrix_cost(pb, neglected_surfs)
    nVarEnd = pl1.numVariablesForPhase(pb["phaseData"][-1])
    
    t1 = clock()
    res = qp.solve_MIP_gurobi_cost(c, nVarEnd, pb["goal"][1],A,b,E,e,linear)
    #res = qpg.solveMIP(c,A,b,E,e)
    t2 = clock()
    
    if res.success:
        res = res.x
    else:
        print ("MIP fail")
        return 1,1,1
    time = 0

    print "time to solve MIP ", time

    ###
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)
    
    #return timMs(t1,t2)
    return pb, res, time
   
   
def solveMIP_gr(pb, surfaces, MIP = True, draw_scene = None, plot = True):  
    if not MIP_OK:
        print "Mixed integer formulation requires gurobi packaged in cvxpy"
        raise ImportError
        
    gurobipy.setParam('LogFile', '')
    gurobipy.setParam('OutputFlag', 0)
       
    A, b, E, e = pl1.convertProblemToLp(pb)   
    c = pl1.slackSelectionMatrix(pb)
    
    ###
    A = A.tolist()
    b = b.tolist()
    E = E.tolist()
    e = e.tolist()
    c = c.tolist()
    
    t1 = clock()
    res = qp.solve_MIP_gurobi(c,A,b,E,e)
    #res = qpg.solveMIP(c,A,b,E,e)
    t2 = clock()
    
    #if res.success:
        #time = res.time
        #res = res.x
    time = res[0]
    del res[0]
    res = array(res)
    print "time to solve MIP ", time
    #else:
        #print ("MIP fail")
        #return 1,1,1
    ###
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)
    
    #return timMs(t1,t2)
    return pb, res, time
        
        
def solveMIP(pb, surfaces, MIP = True, draw_scene = None, plot = True):  
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
    print "time to solve MIP ", timMs(t1,t2)
    
    
    plot = plot and draw_scene is not None 
    if plot:
        ax = draw_scene(surfaces)
        pl1.plotQPRes(pb, res, ax=ax)
    
    #return timMs(t1,t2)
    return pb, res, timMs(t1,t2)
