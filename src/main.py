from sklearn.base import BaseEstimator, ClassifierMixin
from deap import base, creator, gp, tools, algorithms
import operator, math
from sklearn.metrics import auc
import numpy as np
import multiprocessing
from sklearn.metrics import log_loss


def if_then_else(input, output1, output2):
    return output1 if input else output2

def almost_eq(val1, val2, acceptance_range):
	return abs(val1-val2)<acceptance_range

def multiply_fb(val_f, val_b):
	m = 1.0 if val_b else 0.0
	return m*val_f

def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1.0

def protectedPow(base, p):
	if base != 0.0:
		return base**p
	else:
		return 0.0

def lsq_cost_function(y, prediction):
    difference = y-prediction
    difference_sq = np.power(difference,2)
    cost = np.sum(difference_sq)/len(y)
    return cost

def objf_ens_optA(w, Xs, y):
    w = np.abs(w)
    sol = np.zeros(Xs[0].shape)
    for i in range(len(w)):
        sol += Xs[i] * w[i] 
    score = log_loss(y, sol)   
    return score
    
class GeneticProgramming(BaseEstimator, ClassifierMixin):
	primitives_list = [(max, [float, float], float),
		(min, [float, float], float),
		(operator.add, [float, float], float),
		(operator.mul, [float, float], float),
		(operator.sub, [float, float], float),
		(protectedDiv, [float, float], float),
		(protectedPow, [float, float], bool),
		(operator.lt, [float, float], bool),
		(operator.gt, [float, float], bool),
		(multiply_fb, [float, bool], float),
		(almost_eq, [float, float, float], bool),
		(if_then_else, [bool, float, float], float)]
		#self.pset.addPrimitive(operator.xor, [bool, bool], bool)
		#self.pset.addPrimitive(operator.neg, [bool], bool)
		#self.pset.addPrimitive(operator.and_, [bool, bool], bool)
		#self.pset.addPrimitive(operator.or_, [bool, bool], bool)
		#self.pset.addPrimitive(math.cos, 1, [float], [float])
		#self.pset.addPrimitive(math.sin, 1, [float], [float])

	terminals_list = [(3.0, float),
		(2.0, float),
		(0.0, float),
		(1.0, float),
		(-1.0, float),
		(10.0, float),
		(False, bool),
		(True, bool)]


	@staticmethod
	def set_primitives(pset, primitive_list):
		for args in primitive_list:
			pset.addPrimitive(*args)
		return pset

	@staticmethod
	def set_terminals(pset, terminal_list):
		for args in terminal_list:
			pset.addTerminal(*args)
		return pset

	@staticmethod
	def set_tools(toolbox, pset, evalData,
				tournament_size=3,
				tree_mut_min=0,
				tree_mut_max=2,
				tree_min=1,
				tree_max=2):
		toolbox.register("select",tools.selTournament, tournsize=tournament_size),
		toolbox.register("mate",gp.cxOnePoint),
		toolbox.register("expr_mut",gp.genFull, min_=tree_mut_min, max_=tree_mut_max),
		toolbox.register("mutate",gp.mutUniform, expr=toolbox.expr_mut, pset=pset),
		toolbox.register("expr",gp.genHalfAndHalf, pset=pset, min_=tree_min, max_=tree_max),
		toolbox.register("individual",tools.initIterate, creator.Individual, toolbox.expr),
		toolbox.register("population",tools.initRepeat, list, toolbox.individual),
		toolbox.register("compile",gp.compile, pset=pset),
		toolbox.register("evaluate",evalData)
		return toolbox



	def __init__(self,
				input_shape,
				cost_fun=lsq_cost_function,
				pop_size=100,
				terminal_list=terminals_list,
				primitive_list=primitives_list,
				cxpb = 0.5, 
				mutpb = 0.2, 
				ngen = 40,
				tools_dict=None,
				verbose = True,
				n_jobs = 4,                 
				**kwrgs
				):
		self.cost_fun = cost_fun
		self.pop_size = pop_size
		self.pop_size = pop_size
		self.pool = multiprocessing.Pool(n_jobs)
		self.cxpb = cxpb
		self.mutpb = mutpb
		self.ngen = ngen
		self.pset = gp.PrimitiveSetTyped("main", input_shape*[float], float)
		self.pset = self.set_primitives(self.pset, primitive_list)
		self.pset = self.set_terminals(self.pset, terminal_list)
		#self.pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
		self.verbose = verbose
		self.kwrgs = kwrgs
		self.set_algorithm()

	def set_algorithm(self):
		self.toolbox = base.Toolbox()
		#self.toolbox.register("map", self.pool.map)
		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
		creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
		self.toolbox = self.set_tools(self.toolbox, self.pset, self.evalData, **self.kwrgs)
		self.pop = self.toolbox.population(n=self.pop_size)
		self.hof = tools.HallOfFame(1)
		if self.verbose:
			self.stats = tools.Statistics(lambda ind: ind.fitness.values)
			self.stats.register("avg", np.mean)
			self.stats.register("std", np.std)
			self.stats.register("min", np.min) 
			self.stats.register("max", np.max)
		else:
			self.stats=None

	def apply_model_on_set(self, ind, X):
		func = self.toolbox.compile(expr=ind)
		pred_list = [func(*tuple(case)) for case in X]
		prediction = np.array(pred_list)
		return prediction

	def evalData(self, individual):
		prediction = self.apply_model_on_set(individual, self.x)
		result = log_loss(self.y, prediction) 		
		return result,

	def fit(self, x, y):
		self.x = x
		self.y = y
		algorithms.eaSimple(self.pop, self.toolbox, self.cxpb, self.mutpb, self.ngen, self.stats, halloffame=self.hof)

	def predict(self, x):
		best_ind = self.hof[0]
		prediction = self.apply_model_on_set(best_ind, x)
		return prediction

#

class BoostedGeneticProgramming(GeneticProgramming):

	def __init__(self, boosting_rounds=3, *args, **kwargs):
		super(BoostedGeneticProgramming, self).__init__(*args, **kwargs)
		self.boosting_rounds = boosting_rounds
		self.solutions_list = []
		self.set_algorithm()

	def apply_model_on_set(self, ind, X, apply_ind=True):
		prediction = np.zeros( (len(X),) )
		for individual in self.solutions_list:
			prediction += super(BoostedGeneticProgramming, self).apply_model_on_set(individual, X)
		if apply_ind:
			prediction += super(BoostedGeneticProgramming, self).apply_model_on_set(ind, X)
		return prediction

	def fit(self, x, y):
		for i in xrange(self.boosting_rounds):
			super(BoostedGeneticProgramming, self).fit(x,y)
			best_ind = self.hof[0]
			self.solutions_list.append(best_ind)
			self.set_algorithm()

	def predict(self, x):
		prediction = self.apply_model_on_set(None, x, apply_ind=False)
		return prediction

class ForestGeneticProgramming(GeneticProgramming):
	def __init__(self, boosting_rounds=3, *args, **kwargs):
		super(BoostedGeneticProgramming, self).__init__(*args, **kwargs)
		self.boosting_rounds = boosting_rounds
		self.solutions_list = []
		self.set_algorithm()

	def apply_model_on_set(self, ind, X, apply_ind=True):
		prediction = np.zeros( (len(X),) )
		if apply_ind:
			prediction += super(BoostedGeneticProgramming, self).apply_model_on_set(ind, X)
		return prediction

	def ensemble_with_weights(y):
		predictions = []
		for individual in self.solutions_list:
			prediction = super(BoostedGeneticProgramming, self).apply_model_on_set(ind, X)
			prediction.append(prediction)
		bounds = [(0,1)]*len(x0)
		cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
		x0 = np.ones(len(y)) / float(len(predictions)) 
		res = minimize(objf_ens_optA, x0, args=(predictions, y), 
                       method='SLSQP', 
                       bounds=bounds,
                       constraints=cons
                       )
		self.w = res.x

	def fit(self, x, y):
		for i in xrange(self.boosting_rounds):
			super(BoostedGeneticProgramming, self).fit(x,y)
			best_ind = self.hof[0]
			self.solutions_list.append(best_ind)
			self.set_algorithm()
		self.ensemble_with_weights(y)

	def predict(self, x):
		predictions = []
		for individual in self.solutions_list:
			prediction = super(BoostedGeneticProgramming, self).apply_model_on_set(ind, X)
			prediction.append(prediction)
		y_pred = np.zeros(Xs[0].shape)
        for i in range(len(self.w)):
            y_pred += x[i] * self.w[i] 
        return y_pred  